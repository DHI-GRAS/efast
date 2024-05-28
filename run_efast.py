# -*- coding: utf-8 -*-
"""
Created on Mon May 27 10:46:02 2024

@author: rmgu, pase
"""

from datetime import datetime, timedelta
from pathlib import Path
import argparse
from dateutil import rrule

# Import s3_fusion modules
import pyefast.s3_processing as s3
import pyefast.s2_processing as s2
import pyefast.efast as efast


# Test parameters
path = Path("./test_data").absolute()
s3_download_dir = path / "S3/raw"
s3_binning_dir = path / "S3/binning"
s3_composites_dir = path / "S3/composites"
s3_blured_dir = path / "S3/blurred"
s3_calibrated_dir = path / "S3/calibrated"
s3_reprojected_dir = path / "S3/reprojected"
s2_download_dir = path / "S2/raw"
s2_processed_dir = path / "S2/processed"
fusion_dir = path / "fusion_results"


def main(
        start_date: str,
        end_date: str,
        s3_sensor: str,
        s3_bands: list,
        s2_bands: list,
        mosaic_days: int,
        step: int = 5,
):

    # Transform parameters
    start_date = datetime.strptime(start_date, "%Y-%m-%d")
    end_date = datetime.strptime(end_date, "%Y-%m-%d")
    if s3_sensor == "SYN":
        instrument = "SYNERGY"
    else:
        instrument = s3_sensor

    # Create directories if necessary
    for folder in [
        s3_binning_dir,
        s3_composites_dir,
        s3_blured_dir,
        s3_calibrated_dir,
        s3_reprojected_dir,
        s2_processed_dir,
        fusion_dir,
    ]:
        folder.mkdir(parents=True, exist_ok=True)

    # Sentinel-2 pre-processing
    s2.extract_mask_s2_bands(
        s2_download_dir,
        s2_processed_dir
    )
    s2.distance_to_clouds(
        s2_processed_dir
    )
    footprint = s2.get_wkt_footprint(
        s2_processed_dir
    )

    # Sentinel-3 pre-processing
    s3.binning_s3(
        s3_download_dir,
        s3_binning_dir,
        aggregator="mean",
        s3_bands=s3_bands,
        instrument=instrument,
        footprint=footprint,
        snap_memory="24G",
        snap_parallelization=1,
    )
    s3.produce_median_composite(
        s3_binning_dir,
        s3_composites_dir,
        mosaic_days=mosaic_days,
        step=step,
        s3_bands=None,
    )
    s3.smoothing(
        s3_composites_dir,
        s3_blured_dir,
        std=1,
        preserve_nan=False
    )
    s3.reformat_s3(
        s3_blured_dir,
        s3_calibrated_dir,
    )
    s3.reproject_and_crop_s3(
        s3_calibrated_dir,
        s2_processed_dir,
        s3_reprojected_dir
    )

    # Perform EFAST fusion
    for date in rrule.rrule(
        rrule.DAILY,
        dtstart=start_date + timedelta(step),
        until=end_date - timedelta(step),
        interval=step,
    ):
        efast.fusion(
            date,
            s3_reprojected_dir,
            s2_processed_dir,
            fusion_dir,
            product="REFL",
            max_days=100,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-date", default="2023-09-11")
    parser.add_argument("--end-date", default="2023-09-21")
    parser.add_argument("--s3-sensor", default="SYN")
    parser.add_argument("--s3-bands", default=["SDR_Oa04", "SDR_Oa06", "SDR_Oa08", "SDR_Oa17"])
    parser.add_argument("--s2-bands", default=["B02", "B03", "B04", "B8A"])
    parser.add_argument("--mosaic-days", default=100)
    parser.add_argument("--step", required=False, default=5)

    args = parser.parse_args()

    main(
        start_date=args.start_date,
        end_date=args.end_date,
        s3_sensor=args.s3_sensor,
        s3_bands=args.s3_bands,
        s2_bands=args.s2_bands,
        step=args.step,
        mosaic_days=args.mosaic_days,
    )
