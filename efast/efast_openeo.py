from typing import Mapping, Sequence
import openeo
import numpy as np
import scipy


def connect():
    return openeo.connect("https://openeo.dataspace.copernicus.eu/")


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
        s3_bands: Sequence[str] = ["SDR_Oa04", "SDR_Oa06", "SDR_Oa08", "SDR_Oa17"],
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


def extract_mask(cube):
    scl = cube.band("SCL")
    # TODO what are the meanings of these values?
    mask = (scl == 0) | (scl == 3) | (scl > 7)
    return mask


def distance_to_clouds(
    cube: openeo.DataCube, tolerance_percentage=0.05, ratio=30, max_distance=255
):
    udf = openeo.UDF.from_file("efast/distance_transform_udf.py")
    #     kernel_size = np.ceil(max_distance)
    #     gaussian_1d = scipy.signal.windows.gaussian(M=kernel_size, std=255 / 4)
    #     kernel = np.outer(gaussian_1d, gaussian_1d)
    #     kernel /= kernel.sum()

    # dtc = 1 - cube.apply_kernel(kernel)
#     dtc = cube.apply_neighborhood(
#         udf,
#         size=[
#             {"dimension": "x", "value": 61, "unit": "px"},
#             {"dimension": "y", "value": 61, "unit": "px"},
#         ],
#         overlap=[],
#     )
    dtc = cube.apply(udf)
    return dtc
