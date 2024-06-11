# -*- coding: utf-8 -*-
"""
MIT License

Copyright (c) 2024 DHI A/S & contributors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

@author: rmgu, pase
"""

import argparse

from datetime import datetime, timedelta
from pathlib import Path

from creodias_finder import query, download
from dateutil import rrule
import zipfile

import efast.efast as efast
import efast.s2_processing as s2
import efast.s3_processing as s3

# CDSE credentials to download Sentinel-2 and Sentinel-3 imagery
CREDENTIALS = {
    "username": "my-cdse-email",
    "password": "my-cdse-password"
}

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
    aoi_geometry: str,
    s3_sensor: str,
    s3_bands: list,
    s2_bands: list,
    mosaic_days: int,
    step: int,
    cdse_credentials: dict,
    snap_gpt_path: str = "gpt",
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
        s3_download_dir,
        s3_binning_dir,
        s3_composites_dir,
        s3_blured_dir,
        s3_calibrated_dir,
        s3_reprojected_dir,
        s2_processed_dir,
        s2_download_dir,
        fusion_dir,
    ]:
        folder.mkdir(parents=True, exist_ok=True)

    # Download the data from CDSE
    download_from_cdse(
        start_date,
        end_date,
        aoi_geometry,
        s2_download_dir,
        s3_download_dir,
        cdse_credentials)

    # Sentinel-2 pre-processing
    s2.extract_mask_s2_bands(
        s2_download_dir,
        s2_processed_dir,
        bands=s2_bands,
    )
    s2.distance_to_clouds(
        s2_processed_dir,
    )
    footprint = s2.get_wkt_footprint(
        s2_processed_dir,
    )

    # Sentinel-3 pre-processing
    s3.binning_s3(
        s3_download_dir,
        s3_binning_dir,
        footprint=footprint,
        s3_bands=s3_bands,
        instrument=instrument,
        aggregator="mean",
        snap_gpt_path=snap_gpt_path,
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
        preserve_nan=False,
    )
    s3.reformat_s3(
        s3_blured_dir,
        s3_calibrated_dir,
    )
    s3.reproject_and_crop_s3(
        s3_calibrated_dir,
        s2_processed_dir,
        s3_reprojected_dir,
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
            minimum_acquisition_importance=0,
        )


def download_from_cdse(
        start_date,
        end_date,
        aoi_geometry,
        s2_download_dir,
        s3_download_dir,
        credentials):

    # First download Sentinel-3 SYN data
    results = query.query('Sentinel3',
                          start_date=start_date,
                          end_date=end_date,
                          geometry=aoi_geometry,
                          instrument="SYNERGY",
                          productType="SY_2_SYN___",
                          timeliness="NT")
    download.download_list([result['id'] for result in results.values()],
                           outdir=s3_download_dir,
                           threads=3,
                           **credentials)
    for zip_file in s3_download_dir.glob("*.zip"):
        with zipfile.ZipFile(zip_file, "r") as zip_ref:
            zip_ref.extractall(s3_download_dir)

    # Then download Sentinel-2 L2A data
    results = query.query('Sentinel2',
                          start_date=start_date,
                          end_date=end_date,
                          geometry=aoi_geometry,
                          productType="L2A")
    download.download_list([result['id'] for result in results.values()],
                           outdir=s2_download_dir,
                           threads=3,
                           **credentials)
    for zip_file in s2_download_dir.glob("*.zip"):
        with zipfile.ZipFile(zip_file, "r") as zip_ref:
            zip_ref.extractall(s2_download_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-date", default="2023-09-11")
    parser.add_argument("--end-date", default="2023-09-21")
    parser.add_argument("--aoi-geometry", default="POINT (-15.432283 15.402828)")  # Dahra EC tower
    parser.add_argument("--s3-sensor", default="SYN")
    parser.add_argument(
        "--s3-bands", default=["SDR_Oa04", "SDR_Oa06", "SDR_Oa08", "SDR_Oa17"]
    )
    parser.add_argument("--s2-bands", default=["B02", "B03", "B04", "B8A"])
    parser.add_argument("--mosaic-days", default=100)
    parser.add_argument("--step", required=False, default=2)
    parser.add_argument("--cdse-credentials", default=CREDENTIALS)
    parser.add_argument("--snap-gpt-path", required=False, default="gpt")

    args = parser.parse_args()

    main(
        start_date=args.start_date,
        end_date=args.end_date,
        aoi_geometry=args.aoi_geometry,
        s3_sensor=args.s3_sensor,
        s3_bands=args.s3_bands,
        s2_bands=args.s2_bands,
        step=args.step,
        mosaic_days=args.mosaic_days,
        cdse_credentials=args.cdse_credentials,
        snap_gpt_path=args.snap_gpt_path
    )
