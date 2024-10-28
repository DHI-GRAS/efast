import operator
from typing import Tuple
import openeo
import numpy as np
import scipy
from collections.abc import Callable, Mapping, Iterable


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
        s2_bands: Iterable[str] = ["B02", "B03", "B04", "B8A", "SCL"],
        s3_bands: Iterable[str] = [
            "Syn_Oa04_reflectance",
            "Syn_Oa06_reflectance",
            "Syn_Oa08_reflectance",
            "Syn_Oa17_reflectance",
            ],
        temporal_extent: Iterable[str] = ["2022-06-01", "2022-06-30"],
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


def extract_mask(
        cube: openeo.DataCube,
        mask_values: Mapping[str, Iterable[int]],
        *,
        operations: Mapping[Tuple[str, int], Callable],
        ) -> openeo.DataCube:
    """
    Generic method to extract a mask from a data cube.
    Generate a mask that has a value of ``True`` whereever the band specified
    as a key in ``mask_values`` is equal to one of the values speicified
    as a value. Operations other than equality comparison can be specified
    via ``operations``.

        import operator
        extract_mask(
                cube,
                mask_values = {
                    "SCL": (0, 3, 7),
                    },
                operations = {
                    # use ``>`` instead of ``==``
                    # to compare band ``"SCL"`` to value ``7``
                    ("SCL", 7): operator.gt,
                    }
                )

        # scl = cube.band("SCL")
        # mask = (scl == 0) | (scl == 3) | (scl > 7)


    :param cube: The data cube containing at least the band(s) used for masking
    :param mask_values: Mapping of band names to the values that should be masked.
    :param operations: Used to specify a comparison operation different to ``==``
        when comparing bands and values. Operations are applied as ``op(band, value)``
    
    """

    assert(len(mask_values) > 0), "'mask_values' cannot be empty."
    def reduce_single_band(band_name, values_to_mask, mask=None):
        band = cube.band(band_name)
        vm_iter = iter(values_to_mask)

        if mask is None:
            vm_first = next(vm_iter)
            mask = band == vm_first

        for vm in vm_iter:
            op = operations.get((band_name, vm), operator.eq)
            mask |= op(band, vm)

        return mask

    first_band_name, *bands = mask_values.keys()
    first_vm, *values_to_mask = mask_values.values()

    mask = reduce_single_band(first_band_name, first_vm, mask=None)

    for band_name, values_to_mask in zip(bands, values_to_mask):
        mask |= reduce_single_band(band_name, values_to_mask, mask)

    return mask
