from datetime import datetime

import openeo

from openeo import UDF, processes

from efast.constants import S3L2SYNClassificationAerosolFlags, S3L2SYNCloudFlags
from efast.openeo.preprocessing.general import _extract_bit_mask

from . import extract_mask

DATE_FORMAT = "%Y-%m-%d"


def extract_clear_land_mask(s3_cube: openeo.DataCube):
    CLOUD_FLAGS_COMBINED = (
        S3L2SYNCloudFlags.CLOUD
        | S3L2SYNCloudFlags.CLOUD_AMBIGUOUS
        | S3L2SYNCloudFlags.CLOUD_MARGIN
    )
    cloud_mask = (s3_cube.band("CLOUD_flags") & CLOUD_FLAGS_COMBINED.value) != 0
    land_mask = (
        s3_cube.band("SYN_flags") & S3L2SYNClassificationAerosolFlags.SYN_land.value
    ) != 0
    return ((cloud_mask == 0) & land_mask) != 0


# TODO add land mask, broken
# TODO debug as a batch job, to access logs
def _extract_clear_land_mask(s3_cube: openeo.DataCube) -> openeo.DataCube:
    mask = _extract_bit_mask(
        s3_cube,
        {
            "CLOUD_flags": (
                S3L2SYNCloudFlags.CLOUD
                | S3L2SYNCloudFlags.CLOUD_AMBIGUOUS
                | S3L2SYNCloudFlags.CLOUD_MARGIN
            ).value
        },
        invert=True,
    )  # | extract_bit_mask(
    #    s3_cube,
    #    {
    #        "SYN_flags": S3L2SYNClassificationAerosolFlags.SYN_land.value,
    #    },
    # )
    return mask


def composite(cube):
    # What we want: sliding window computation
    # distance_to_cloud_score = compute_distance_to_cloud_score(distance_to_cloud)  # add normalization
    # merged = distance_to_cloud_score.merge_cubes(cube)
    # merged.apply_neighbourhood(process=produce_weighted_composite)
    pass


# TODO rename D to something more useful
def compute_distance_to_cloud_score(
    distance_to_cloud: processes.ProcessBuilder, D: float
) -> processes.ProcessBuilder:
    score = (distance_to_cloud - 1) / D
    return processes.clip(score, 0.0, 1.0)


def compute_time_weighted_cube(
    unweighted_cube: processes.ProcessBuilder, target_date: datetime
) -> processes.ProcessBuilder:
    udf_compute_temporal_score = UDF.from_file(
        "efast/openeo/udf/compute_temporal_score",
        context={"target_date": target_date.strftime(DATE_FORMAT)},
    )
    return unweighted_cube.apply_dimension(
        process=udf_compute_temporal_score, dimension="t"
    )


def produce_weighted_composite(
    s3_cube: processes.ProcessBuilder,
    distance_to_cloud_score: processes.ProcessBuilder,
    target_date: datetime,
) -> processes.ProcessBuilder:
    """
    :param distance_to_cloud_score:
    """
    # input1: all relevant Sentinel-3 images to produce a single output composite
    # input2: distance to cloud scores for all Sentinel-3 observations to avoid
    # recomputing them for each composite

    pass
