from typing import Mapping, Sequence
import openeo
import numpy as np
import scipy


def connect():
    return openeo.connect("https://openeo.dataspace.copernicus.eu/")


# TODO move somewhere else, not specific to sentinel-2
class TestArea:
    # aoi_wkt = "POINT (-15.432283 15.402828)"
    directions = ["west", "south", "east", "north"]
    bbox_list = [-15.456047, 15.665024, -15.425491, 15.687501]
    bbox = {d: c for (d, c) in zip(directions, bbox_list)}

    def __init__(
        self,
        *,
        bbox: Mapping[str, float] = bbox,
        s2_bands: Sequence[str] = ["B02", "B03", "B04", "B8A", "SCL"],
        s3_bands: Sequence[str] = [
            "Syn_Oa04_reflectance",
            "Syn_Oa06_reflectance",
            "Syn_Oa08_reflectance",
            "Syn_Oa17_reflectance",
            ],
        temporal_extent: Sequence[str] = ["2022-06-01", "2022-06-30"],
    ) -> None:
        self.bbox = bbox
        self.s2_bands = s2_bands
        self.s3_bands = s3_bands
        self.temporal_extent = temporal_extent

    def get_s2_cube(self, connection):
        return connection.load_collection(
            "SENTINEL2_L2A",
            spatial_extent=self.bbox,
            temporal_extent=self.temporal_extent,
            bands=self.s2_bands,
        )

    def get_s3_cube(self, connection):
        return connection.load_collection(
            "SENTINEL3_SYN_L2_SYN",
            spatial_extent=self.bbox,
            temporal_extent=self.temporal_extent,
            bands=self.s3_bands,
        )


def extract_cloud_mask(cube: openeo.DataCube):
    scl = cube.filter_bands(["SCL"])
    # 0: No data
    # 3: Cloud shadow
    # 8-10: Clouds
    # 11: Snow or ice

    mask = (scl == 0) | (scl == 3) | (scl > 7)
    return mask


def calculate_large_grid_cloud_mask(cube: openeo.DataCube, tolerance_percentage: float = 0.05, grid_side_length: int=300):
    cloud_mask = extract_cloud_mask(cube)
    # FIXME check also if there is negative or zero data, otherwise results will differ
    
    # TODO this could better be resample_cube_spatial, because we are matching to a sentinel-3 cube
    cloud_mask = cloud_mask * 1.0 # convert to float
    cloud_mask_resampled = cloud_mask.resample_spatial(grid_side_length, method="average") # resample to sentinel-3 size

    # TODO extract UDF to file
    # UDF to apply an element-wise less than operation. Normal "<" does not properly work on openEO datacubes
    udf = openeo.UDF(f"""
import numpy as np
import xarray as xr
def apply_datacube(cube: XarrayDataCube, context: dict) -> XarrayDataCube:
    array = cube.get_array()
    array = array < {tolerance_percentage}
    #return XarrayDataCube(xr.DataArray(array, dims=["t", "x", "y", "bands"]))
    return XarrayDataCube(xr.DataArray(array, dims=["bands", "x", "y"]))
            """)
    return cloud_mask_resampled.apply(process=udf)


def distance_to_clouds(
    cube: openeo.DataCube, tolerance_percentage=0.05, ratio=30, max_distance=255
):
    return _distance_to_clouds_udf(cube, tolerance_percentage=tolerance_percentage, ratio=ratio, max_distance=max_distance)


def _distance_to_clouds_kernel(
    cube: openeo.DataCube, tolerance_percentage=0.05, ratio=30, max_distance=7
):
    kernel_size = np.ceil(max_distance)
    gaussian_1d = scipy.signal.windows.gaussian(M=kernel_size, std=255 / 4)
    kernel = np.outer(gaussian_1d, gaussian_1d)
    kernel /= kernel.sum()

    dtc = cube.apply_kernel(kernel)
    return dtc


# TODO implement max_distance as a parameter to the UDF
# TODO replace hard coded tile size (366)
def _distance_to_clouds_udf(
    cube: openeo.DataCube, tolerance_percentage=0.05, ratio=30, max_distance=255
):
    udf = openeo.UDF.from_file("efast/distance_transform_udf.py")
    dtc = cube.apply_neighborhood(
        udf,
        size=[
            {"dimension": "x", "value": 366, "unit": "px"},
            {"dimension": "y", "value": 366, "unit": "px"},
        ],
        overlap=[],
    )
    return dtc
