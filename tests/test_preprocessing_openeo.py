import shutil
import time
import logging

from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import rasterio

from openeo.local import LocalConnection
from openeo.udf import execute_local_udf, XarrayDataCube
from rasterio import warp

from efast import efast_openeo, s2_processing

TEST_DATA_ROOT = Path(__file__).parent.parent / "test_data"
TEST_DATA_S2 = TEST_DATA_ROOT / "S2"

TEST_DATA_S2_NETCDF = (
    TEST_DATA_S2
    / "netcdf"
    / "S2B_MSIL2A_20220603T112119_N0400_R037_T28PDC_20220603T140337_s2resampled.nc"
)


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
    conn = LocalConnection([TEST_DATA_S2 / "netcdf"])
    assert len(conn.list_collections()) > 0, "No data found in local collection"

    with create_temp_dir_and_copy_files(TEST_DATA_S2, sub="raw/", pattern=None) as tmp:
        tmp_path = Path(tmp.name)
        tif_path = tmp_path / "tif"
        tif_path.mkdir()

        # reference
        s2_processing.extract_mask_s2_bands(TEST_DATA_S2 / "raw", tif_path)
        s2_processing.distance_to_clouds(tif_path, ratio=30, tolerance_percentage=0.05)
        try:
            reference_path = next(tif_path.glob("*20220608*DIST_CLOUD.tif"))
        except StopIteration:
            raise RuntimeError(
                "Tif path does not contain any tifs after extracting bands"
            )
        with rasterio.open(reference_path, "r") as ds:
            dtc_reference = ds.read(1)
            bounds = transform_bounds_to_epsg_4326(ds.crs, ds.bounds)

        # make smaller

        scale = 3
        rows, columns = map(lambda x: x // scale, dtc_reference.shape)
        bounds["west"] = bounds["east"] - (bounds["east"] - bounds["west"]) / scale
        bounds["south"] = bounds["north"] + (bounds["south"] - bounds["north"]) / scale

        # openeo

        conn = efast_openeo.connect()
        conn.authenticate_oidc()

        test_area = efast_openeo.TestArea(bbox=bounds, s2_bands=["SCL"], temporal_extent=("2022-06-08", "2022-06-08"))
        cube = test_area.get_s2_cube(conn)

        scl = cube.filter_bands("SCL")
        cloud_mask = (scl == 0) | (scl == 3) | (scl > 7)

        # TODO this could better be resample_cube_spatial, because we are matching to a sentinel-3 cube
        cloud_mask_resampled = cloud_mask.resample_spatial(300, method="average") # resample to sentinel-3 size
        tol = 0.05
        dtc = efast_openeo.distance_to_clouds(cloud_mask_resampled < tol, tolerance_percentage=0.05, ratio=30)
        download_path = tmp_path / "test_distance_to_cloud.nc"
        before = time.perf_counter()

        print("openEO execution")

        dtc.download(download_path)
        elapsed = time.perf_counter() - before
        print(f"executed and downloaded in {elapsed:.2f}s")
        with rasterio.open(download_path, "r") as ds:
            dtc_openeo = ds.read(1)

        assert dtc_openeo.shape >= (rows, columns)
        assert dtc_reference.shape >= (rows, columns)

        assert np.all(dtc_reference[:rows, -columns:] == dtc_openeo[:rows, -columns:])


def test_dtc():
    cube = np.zeros((61, 61))
    cube[0, 0] = 1

    efast_openeo.distance_to_clouds(cube, max_distance=15)

    # Notes
    # Try transforming to a boolean (0 or 1) array before passing it to dtc


def transform_bounds_to_epsg_4326(crs, bounding_box):
    bounds = warp.transform_bounds(
        crs,
        "EPSG:4326",
        bounding_box.left,
        bounding_box.bottom,
        bounding_box.right,
        bounding_box.top,
    )
    directions = ["west", "south", "east", "north"]

    return {d:b for (d,b) in zip(directions, bounds)}
