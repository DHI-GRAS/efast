from pathlib import Path
import numpy as np
import rasterio
from rasterio.transform import Affine
import time
from efast import eoprofiling

from efast.binning import FILL_VALUE, BBox, SynProduct, bin_to_grid, create_geogrid, get_reflectance_filename, bin_to_grid_numpy
from efast.constants import S3L2SYNClassificationAerosolFlags, S3L2SYNCloudFlags

SYN_PRODUCT_PATH = (
    Path(__file__).parent.parent
    / "test_data"
    / "S3"
    / "raw"
    / "S3A_SY_2_SYN____20220601T105211_20220601T105511_20220603T070140_0179_086_051_2700_PS1_O_NT_002.SEN3"
)
REGION = "POLYGON ((-14.909092758013342 15.288158060474164, -14.908649214991065 16.28081360439369, -15.936300787396918 16.27876184113824, -15.931755484482125 15.286237505126845, -14.909092758013342 15.288158060474164))"

TEST_DATA_OUTPUT_PATH = Path(__file__).parent.parent / "test_temporary_data"

def test_s3reader_dev():
    band_names = [get_reflectance_filename(i) for i in (4, 6, 8, 17)]
    band_names.append("CLOUD_flags")
    prod = SynProduct(SYN_PRODUCT_PATH, band_names=band_names)
    prod.read_bands()

    return True


def test_create_geogrid():
    lat, lon = create_geogrid(BBox(lat_min=0, lat_max=30, lon_min=0, lon_max=30))
    return True

CLOUD_FLAGS_COMBINED = (
    S3L2SYNCloudFlags.CLOUD
    | S3L2SYNCloudFlags.CLOUD_AMBIGUOUS
    | S3L2SYNCloudFlags.CLOUD_MARGIN
)

def test_bin_to_grid_dev():
    reflectance_bands = [get_reflectance_filename(i) for i in (4, 6, 8, 17)]
    band_names = [*reflectance_bands, "CLOUD_flags", "SYN_flags"]
    prod = SynProduct(SYN_PRODUCT_PATH, band_names=band_names)
    ds = prod.read_bands()
    bbox = BBox.from_wkt(REGION)
    grid = create_geogrid(bbox, num_rows=66792)

    cloud_no_land_mask = ((ds["CLOUD_flags"] & CLOUD_FLAGS_COMBINED.value) != 0) & (ds["SYN_flags"] & S3L2SYNClassificationAerosolFlags.SYN_land.value > 0)

    for reflectance_band in reflectance_bands:
         band_values = ds[reflectance_band].data
         band_values[cloud_no_land_mask] = -10000
         ds[reflectance_band] = (["lat", "lon"], band_values)


    binned = bin_to_grid_numpy(ds, reflectance_bands, grid, super_sampling=2)
    pixel_size = grid.lat[1] - grid.lat[0]

    # TODO
    binned[binned < 0] = np.nan

    transform = Affine.translation(grid.lon[0], grid.lat[-1]) * Affine.scale(pixel_size, pixel_size)

    with rasterio.open(TEST_DATA_OUTPUT_PATH / "binned.tif", "w", driver="GTiff", height=binned.shape[1], width=binned.shape[2], count=len(reflectance_bands), dtype=binned.dtype, crs="+proj=latlong", transform=transform, nodata=np.nan) as ds:
        ds.write(binned[:, ::-1, :])

    return True

def test_bin_to_grid_dev_timed():
    with eoprofiling.Profiling(TEST_DATA_OUTPUT_PATH / "prof.out"):
        start = time.perf_counter()
        test_bin_to_grid_dev()
        elapsed = time.perf_counter() - start

        print(f"test ran in {elapsed:.2f} seconds")

    assert True

def test_bin_to_grid_zoomed_dev():
    band_names = [get_reflectance_filename(i) for i in (4, 6, 8, 17)]
    band_names.append("CLOUD_flags")
    prod = SynProduct(SYN_PRODUCT_PATH, band_names=band_names)
    ds = prod.read_bands()
    bbox = BBox.from_wkt(REGION)
    grid = create_geogrid(bbox, num_rows=66792)
    res = bin_to_grid(ds, band_names, grid, super_sampling=2, interpolation_order=1)
    stats = res.statistic

    pixel_size = grid.lat[1] - grid.lat[0]
    # TODO why last lat?
    transform = Affine.translation(grid.lon[0], grid.lat[-1]) * Affine.scale(pixel_size, pixel_size)

    with rasterio.open(TEST_DATA_OUTPUT_PATH / "binned_super_sampled.tif", "w", driver="GTiff", height=stats.shape[0], width=stats.shape[1], count=1, dtype=stats.dtype, crs="+proj=latlong", transform=transform) as ds:
        ds.write(stats[::-1, :], 1)

    return True

if __name__ == "__main__":
    test_bin_to_grid_dev_timed()
