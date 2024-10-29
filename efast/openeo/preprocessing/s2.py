import operator

import openeo

from efast.constants import S2L2ASCFlags

from . import extract_mask


def extract_cloud_mask(cube: openeo.DataCube) -> openeo.DataCube:
    return extract_mask(
        cube,
        {
            "SCL": [
                S2L2ASCFlags.NO_DATA.value,
                S2L2ASCFlags.CLOUD_SHADOWS.value,
                S2L2ASCFlags.UNCLASSIFIED.value,
            ]
        },
        operations={
            ("SCL", 7): operator.gt, # consider flags higher than 7 as clouds
        },
    )


def calculate_large_grid_cloud_mask(
    cube: openeo.DataCube,
    tolerance_percentage: float = 0.05,
    grid_side_length: int = 300,
):
    cloud_mask = extract_cloud_mask(cube)
    # FIXME check also if there is negative or zero data, otherwise results will differ

    # TODO this could better be resample_cube_spatial, because we are matching to a sentinel-3 cube
    cloud_mask = cloud_mask * 1.0  # convert to float
    cloud_mask_resampled = cloud_mask.resample_spatial(
        grid_side_length, method="average"
    )  # resample to sentinel-3 size
    return cloud_mask_resampled < tolerance_percentage
