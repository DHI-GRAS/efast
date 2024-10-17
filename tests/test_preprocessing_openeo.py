import shutil
import time

from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import xarray as xr
import rasterio

import openeo
from openeo.udf import execute_local_udf

from efast import s2_processing_openeo, s2_processing

TEST_DATA_ROOT = Path(__file__).parent.parent / "test_data"
TEST_DATA_S2 = TEST_DATA_ROOT / "S2"

TEST_DATA_S2_NETCDF = (
    TEST_DATA_S2
    / "netcdf"
    / "S2B_MSIL2A_20220603T112119_N0400_R037_T28PDC_20220603T140337_s2resampled.nc"
)

TEST_DATE = "20220618"
TEST_DATE_DASH = "2022-06-18"

@contextmanager
def create_temp_dir_and_copy_files(source, sub=".", pattern: str | None = "*"):
    import sys

    tmp = TemporaryDirectory()
    if pattern is not None:
        for f in Path(source).glob(pattern):
            print("copying ", f.name, file=sys.stderr)
            shutil.copytree(f, Path(tmp.name) / sub / f.name)
    try:
        yield tmp
    finally:
        tmp.cleanup()


def test_distance_to_cloud():
    with create_temp_dir_and_copy_files(TEST_DATA_S2, sub="raw/", pattern=None) as tmp:
        tmp_path = Path(tmp.name)
        tif_path = tmp_path / "tif"
        tif_path.mkdir()

        # reference

        s2_processing.extract_mask_s2_bands(TEST_DATA_S2 / "raw", tif_path, resolution=20)
        s2_processing.distance_to_clouds(tif_path, ratio=15, tolerance_percentage=0.05)
        try:
            reference_path = next(tif_path.glob(f"*{TEST_DATE}*DIST_CLOUD.tif"))
        except StopIteration:
            raise RuntimeError(
                "Tif path does not contain any tifs after extracting bands"
            )
        with rasterio.open(reference_path, "r") as ds:
            dtc_reference = ds.read(1)
            bounds = spatial_extent_from_bounds(ds.crs, bounding_box=ds.bounds)

        # FIXME we need the rows and columns only because there is an issue with openeo that causes
        # the results to have a different grid than they should.
        # See https://forum.dataspace.copernicus.eu/t/resample-spatial-does-not-properly-align/1278
        rows, columns = dtc_reference.shape

        # openeo

        conn = s2_processing_openeo.connect()
        conn.authenticate_oidc()

        test_area = s2_processing_openeo.TestArea(bbox=bounds, s2_bands=["SCL"], temporal_extent=(TEST_DATE_DASH, TEST_DATE_DASH))
        cube = test_area.get_s2_cube(conn)

        dtc_input = s2_processing_openeo.calculate_large_grid_cloud_mask(cube, tolerance_percentage=0.05, grid_side_length=300)
        dtc = s2_processing_openeo.distance_to_clouds(dtc_input, tolerance_percentage=0.05, ratio=30)
        download_path = tmp_path / "test_distance_to_cloud.tif"

        print("openEO execution")

        # intermediate results for debugging
        BASE_DIR = Path("openeo_results")
        #BASE_DIR.mkdir(exist_ok=True)
        #cloud_mask.download(BASE_DIR / "cloud_mask.tif")
        #cloud_mask_resampled.download(BASE_DIR / "cloud_mask_resampled.tif")
        #dtc_input.download(BASE_DIR / "dtc_input.tif")

        before = time.perf_counter()
        dtc.download(download_path)
        shutil.copy(download_path, BASE_DIR)
        elapsed = time.perf_counter() - before
        print(f"executed and downloaded in {elapsed:.2f}s")

        with rasterio.open(download_path, "r") as ds:
            dtc_openeo = ds.read(1)

        assert dtc_openeo.shape >= (rows, columns)
        assert dtc_reference.shape >= (rows, columns)

        assert np.all(dtc_reference[:rows, -columns:] == dtc_openeo[:rows, -columns:])


def test_distance_to_cloud_synthetic_cube():
    cube = np.zeros((60, 60))
    cube[:30, :30] = 1
    ratio = 30
    tolerance = 0.05
    cube_resampled = (
        (cube == 0)
        .reshape(cube.shape[0] // ratio, ratio, cube.shape[1] // ratio, ratio)
        .mean(3)
        .mean(1)
    ) < tolerance
    cube = xr.DataArray(cube, dims=["x", "y"])
    cube_resampled = xr.DataArray(cube_resampled, dims=["x", "y"])

    udf = openeo.UDF.from_file("efast/distance_transform_udf.py")
    dtc_local_udf = execute_local_udf(udf, cube_resampled).get_datacube_list()[0].get_array()

    dtc_reference = s2_processing.distance_to_clouds_in_memory(cube.to_numpy(), ratio=ratio, tolerance_percentage=tolerance)

    assert np.all(dtc_reference == dtc_local_udf)


def spatial_extent_from_bounds(crs, bounding_box):
    directions = ["west", "south", "east", "north"]

    extent = {d:b for (d,b) in zip(directions, bounding_box)}
    extent["crs"] = extract_epsg_code_from_rasterio_crs(crs)
    return extent


def extract_epsg_code_from_rasterio_crs(crs: rasterio.CRS) -> int:
    code_str = crs["init"].split(":")[1]
    return int(code_str)

