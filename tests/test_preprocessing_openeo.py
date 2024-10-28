from collections.abc import Generator
import shutil
import time
import pytest

from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import openeo
import rasterio
import xarray as xr
import pyproj
import shapely
from shapely import wkt
from enhancement_tools.time_measurement import Timer

from openeo.udf import execute_local_udf

from efast.openeo import preprocessing
# TODO this should live somewhere else
from efast.openeo.preprocessing import connect
from efast import s2_processing, s3_processing


TEST_DATA_ROOT = Path(__file__).parent.parent / "test_data"
TEST_DATA_S2 = TEST_DATA_ROOT / "S2"

TEST_DATA_S3 = TEST_DATA_ROOT / "S3"

TEST_DATA_S2_NETCDF = (
    TEST_DATA_S2
    / "netcdf"
    / "S2B_MSIL2A_20220603T112119_N0400_R037_T28PDC_20220603T140337_s2resampled.nc"
)

TEST_DATE = "20220618"
TEST_DATE_DASH = "2022-06-18"

SKIP_LOCAL = True
DOWNLOAD_INTERMEDIATE_RESULTS = True
SMALL_AREA = True


@contextmanager
def create_temp_dir_and_copy_files(
    source, sub=".", pattern: str | None = "*"
) -> Generator[TemporaryDirectory]:
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


############################## S2 Processing ##############################


@pytest.mark.skip
def test_distance_to_cloud():
    with create_temp_dir_and_copy_files(TEST_DATA_S2, sub="raw/", pattern=None) as tmp:
        tmp_path = Path(tmp.name)
        tif_path = tmp_path / "tif"
        tif_path.mkdir()

        # reference

        s2_processing.extract_mask_s2_bands(
            TEST_DATA_S2 / "raw", tif_path, resolution=20
        )
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

        conn = connect()
        conn.authenticate_oidc()

        test_area = preprocessing.TestArea(
            bbox=bounds,
            s2_bands=["SCL"],
            temporal_extent=(TEST_DATE_DASH, TEST_DATE_DASH),
        )
        cube = test_area.get_s2_cube(conn)

        dtc_input = preprocessing.s2.calculate_large_grid_cloud_mask(
            cube, tolerance_percentage=0.05, grid_side_length=300
        )
        dtc = preprocessing.distance_to_clouds(
            dtc_input, tolerance_percentage=0.05, ratio=30
        )
        download_path = tmp_path / "test_distance_to_cloud.tif"

        print("openEO execution")
        before = time.perf_counter()
        #dtc.download(download_path)
        elapsed = time.perf_counter() - before
        print(f"executed and downloaded in {elapsed:.2f}s")

        if DOWNLOAD_INTERMEDIATE_RESULTS:
            BASE_DIR = Path("openeo_results")
            BASE_DIR.mkdir(exist_ok=True)
            print("downloading input")  # TMP
            (dtc_input * 1.0).download(BASE_DIR / "dtc_input.tif")
            #shutil.copy(download_path, BASE_DIR)
        print("downloading result") # TMP
        dtc.download(download_path) # TMP
        shutil.copy(download_path, BASE_DIR) # TMP


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
    #cube = cube.add_dimension(name="bands", label="mask", type="bands")
    cube_resampled = xr.DataArray(cube_resampled[np.newaxis, np.newaxis], dims=["bands", "t", "x", "y"])

    udf = openeo.UDF.from_file("efast/distance_transform_udf.py")
    dtc_local_udf = (
        execute_local_udf(udf, cube_resampled).get_datacube_list()[0].get_array()
    )

    dtc_reference = s2_processing.distance_to_clouds_in_memory(
        cube.to_numpy(), ratio=ratio, tolerance_percentage=tolerance
    )

    assert np.all(dtc_reference == dtc_local_udf)


def spatial_extent_from_bounds(crs, bounding_box):
    directions = ["west", "south", "east", "north"]

    extent = {d: b for (d, b) in zip(directions, bounding_box)}
    extent["crs"] = extract_epsg_code_from_rasterio_crs(crs)
    return extent


def extract_epsg_code_from_rasterio_crs(crs: rasterio.CRS) -> int:
    code_str = crs["init"].split(":")[1]
    return int(code_str)


############################## S3 Processing ##############################

# TODO this should not live in test code
CLOUD_FLAG_CLOUD = 0b1
CLOUD_FLAG_CLOUD_AMBIGUOUS = 0b10
CLOUD_FLAG_CLOUD_MARGIN = 0b100
CLOUD_FLAGS_COMBINED = (
    CLOUD_FLAG_CLOUD | CLOUD_FLAG_CLOUD_AMBIGUOUS | CLOUD_FLAG_CLOUD_MARGIN
)
OLC_FLAG_LAND = 0b1 << 12


def test_data_acquisition_s3():
    with create_temp_dir_and_copy_files(
        TEST_DATA_S3, sub="raw/", pattern=f"raw/*SY_2_SYN____2022061*"
        #TEST_DATA_S3, sub="raw/", pattern=f"raw/*SY_2_SYN____*"
    ) as tmp:
        inner_data_acquisition_s3(tmp)


def inner_data_acquisition_s3(tmpdir):
    tmp = Path(tmpdir.name)
    # TODO for interactive testing
    download_dir = Path("temp_s3")
    download_dir.mkdir(exist_ok=True)

    bounds = {
        "west": 399960.0,
        "south": 1590240.0,
        "east": 509760.0,
        "north": 1700040.0,
        "crs": 32628,
    }

    if SMALL_AREA:
        dist = 3600
        bounds["east"] = bounds["west"] + dist
        bounds["north"] = bounds["south"] + dist

    # reference

    s3_binning_dir = tmp / "binning"
    s3_composites_dir = tmp / "composites"
    s3_download_dir = tmp / "raw"

    s3_binning_dir.mkdir()
    s3_composites_dir.mkdir()
    footprint = transform_bounds_to_wkt(
        bounds
    )  # TODO probably needs to be converted to wkt

    if not SKIP_LOCAL:
        s3_processing.binning_s3(
            s3_download_dir,
            s3_binning_dir,
            footprint=footprint,
            s3_bands=["SDR_Oa04", "SDR_Oa06", "SDR_Oa08", "SDR_Oa17"],
            instrument="SYNERGY",
            max_zenith_angle=30,
            crs="EPSG:32628",
            aggregator="mean",
            snap_gpt_path="gpt",
            snap_memory="8G",
            snap_parallelization=1,  # TODO more than 1?
        )

        s3_processing.produce_median_composite(
            s3_binning_dir,
            s3_composites_dir,
            mosaic_days=100,
            step=2,
            s3_bands=None,
        )

        shutil.copytree(s3_binning_dir, download_dir, dirs_exist_ok=True)

    # openeo

    conn = connect()
    conn.authenticate_oidc()

    bands = [
        # "Syn_Oa04_reflectance",
        # "Syn_Oa06_reflectance",
        # "Syn_Oa08_reflectance",
        "Syn_Oa17_reflectance",
        "CLOUD_flags",
        "OLC_flags",
    ]
    test_area = preprocessing.TestArea(
        bbox=bounds, s3_bands=bands, temporal_extent=(TEST_DATE_DASH, TEST_DATE_DASH)
    )
    cube = test_area.get_s3_cube(conn)
    cloud_flags = cube.band("CLOUD_flags")
    olc_flags = cube.band("OLC_flags")

    cloud_mask = (cloud_flags & CLOUD_FLAGS_COMBINED) != 0
    land_mask = (olc_flags & OLC_FLAG_LAND) != 0
    mask = ((~cloud_mask) & land_mask) == 0

    cube_masked = cube.mask(mask)
    mask = mask.add_dimension(name="bands", label="cloud_noland_mask", type="bands")
    merged = cube_masked.merge_cubes(mask)

    with Timer(dsc="OpenEO execution and download") as timer:
        # cube.download(download_dir / "cube_unmasked.nc")
        merged.download(download_dir / "cube_masked.nc")

    print(f"execution and download in {timer.elapsed:.2f} seconds")

    assert True, "Test ran through without issues"
    return True


def transform_bounds_to_wkt(bounds: dict):
    required = ["crs", "west", "east", "south", "north"]
    for att in required:
        if att not in bounds:
            raise ValueError(f"'bounds' needs to contain all of {required}")
    transformer = pyproj.Transformer.from_crs(
        f"EPSG:{bounds['crs']}", "EPSG:4326", always_xy=True
    )
    minx_wgs84, miny_wgs84 = transformer.transform(bounds["west"], bounds["south"])
    maxx_wgs84, maxy_wgs84 = transformer.transform(bounds["east"], bounds["north"])
    bbox = shapely.geometry.box(
        minx=minx_wgs84, miny=miny_wgs84, maxx=maxx_wgs84, maxy=maxy_wgs84
    )
    return wkt.dumps(bbox)

METADATA_UDF = openeo.UDF("""
import numpy as np
import xarray as xr
from openeo.udf import XarrayDataCube
def apply_datacube(cube: XarrayDataCube, context: dict) -> XarrayDataCube:
    array = cube.get_array()
    array = array < {tolerance_percentage}
    #return XarrayDataCube(xr.DataArray(array, dims=["t", "x", "y", "bands"]))
    return XarrayDataCube(xr.DataArray(array, dims=["bands", "x", "y"]))
""")
