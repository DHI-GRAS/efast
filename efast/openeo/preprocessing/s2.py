import operator

import openeo

from . import extract_mask


def extract_cloud_mask_s2(cube: openeo.DataCube) -> openeo.DataCube:
    return extract_mask(
        cube,
        {"SCL": [0, 3, 7]},
        operations={
            ("SCL", 7): operator.gt,
        },
    )

def calculate_large_grid_cloud_mask(cube: openeo.DataCube, tolerance_percentage: float = 0.05, grid_side_length: int=300):
    cloud_mask = extract_cloud_mask_s2(cube)
    # FIXME check also if there is negative or zero data, otherwise results will differ
    
    # TODO this could better be resample_cube_spatial, because we are matching to a sentinel-3 cube
    cloud_mask = cloud_mask * 1.0 # convert to float
    cloud_mask_resampled = cloud_mask.resample_spatial(grid_side_length, method="average") # resample to sentinel-3 size
    return cloud_mask_resampled < tolerance_percentage
