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
import zipfile
import os
import logging

from typing import Tuple
from datetime import datetime, timedelta
from pathlib import Path

from enhancement_tools.time_measurement import Timer
from creodias_finder import download, query
from dateutil import rrule
from tqdm import tqdm

from efast import binning
import efast.efast as efast
import efast.s2_processing as s2
import efast.s3_processing as s3

SKIP_CREDENTIALS = True

# CDSE credentials to download Sentinel-2 and Sentinel-3 imagery
def get_credentials_from_env():
    username = os.getenv("CDSE_USER")
    password = os.getenv("CDSE_PASSWORD")
    if not SKIP_CREDENTIALS and (username is None or password is None):
        raise RuntimeError(
                "Please make sure that CDSE credentials are available in the environment.\n"
                "In order to proceed, set 'CDSE_USER' and 'CDSE_PASSWORD' accordingly."
        )

    print("loaded credentials for user", username)
    return {
        "username": username,
        "password": password
    }

# TODO remove when downloads already happened
# CDSE credentials to download Sentinel-2 and Sentinel-3 imagery
CREDENTIALS = get_credentials_from_env()

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

MARKERS = Path("./markers/")
DOWNLOAD_MARKER = MARKERS / "download.txt"
S2_PRE_MARKER = MARKERS / "s2_pre.txt"
S3_PRE_MARKER = MARKERS / "s3_pre.txt"


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
    ratio: int,
    snap_gpt_path: str = "gpt",
    use_snap_binning: bool = False,
):
    logger = logging.getLogger(__file__)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt='%Y-%m-%d %I:%M:%S')
    timings: list[Tuple[str, float]] = []

    logger.info(f"Creating marker directory at {MARKERS.resolve()}")
    MARKERS.mkdir(exist_ok=True, parents=True)

    logger.info(f"Creating output directories ({fusion_dir}, ...)")
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

    if not DOWNLOAD_MARKER.exists():
        logger.info(f"Downloading products from CDSE for AOI: {aoi_geometry}, between {start_date} and {end_date}")
        instrument = download_files(
            start_date=start_date,
            end_date=end_date,
            aoi_geometry=aoi_geometry,
            s3_sensor=s3_sensor,
            cdse_credentials=cdse_credentials,
        )
        logger.info("Download complete")
        with open(DOWNLOAD_MARKER, "w") as fp:
            logger.debug(f"Writing instrument {instrument} to {DOWNLOAD_MARKER}")
            fp.write(instrument)
    else:
        with open(DOWNLOAD_MARKER, "r") as fp:
            instrument = fp.readline()
        logger.info(f"No downloads performed, because {DOWNLOAD_MARKER} exists. Read instrument '{instrument}' from marker file.")

    if not S2_PRE_MARKER.exists():
        logger.info("Performing pre-processing on Sentinel-2 images")

    
        with Timer(dsc="Sentinel-2 preprocessing", logger=logger) as t:
            footprint = s2_preprocessing(s2_bands=s2_bands, ratio=ratio)
        timings.append(("Sentinel-2 pre-processing", t.elapsed))

        with open(S2_PRE_MARKER, "w") as fp:
            logger.debug(f"Writing footprint {footprint} to {S2_PRE_MARKER}")
            fp.write(footprint)
    else:
        with open(S2_PRE_MARKER, "r") as fp:
            footprint = fp.readline()
        logger.info(f"No Sentinel-2 pre-processing performed, {S2_PRE_MARKER} exists. Read footprint {footprint} from marker file.")

    if not S3_PRE_MARKER.exists():
        logger.info("Performing pre-processing on Sentinel-3 images")
        with Timer(dsc="Sentinel-3 preprocessing", logger=logger) as t:
            s3_preprocessing(
                footprint=footprint,
                s3_bands=s3_bands,
                instrument=instrument,
                snap_gpt_path=snap_gpt_path,
                mosaic_days=mosaic_days,
                step=step,
                use_snap_binning=use_snap_binning,
            )
        timings.append(("Sentinel-3 pre-processing", t.elapsed))

        with open(S3_PRE_MARKER, "w") as fp:
            logger.debug(f"Creating marker file {S3_PRE_MARKER}.")
            fp.write("mark")
    else:
        logger.info(f"No Sentinel-3 pre-processing performed, {S3_PRE_MARKER} exists.")

    date_format = "%Y-%m-%d"
    logger.info(f"Performing EFAST for {datetime.strptime(start_date, date_format)} to {datetime.strptime(end_date, date_format)}")


    with Timer(dsc="EFAST", logger=logger) as t:
        perform_efast(start_date=datetime.strptime(start_date, date_format), end_date=datetime.strptime(end_date, date_format), step=step, ratio=ratio)

    timings.append(("EFAST", t.elapsed))

    for desc, t in timings:
        logger.info(f"[Timings] {t} ({desc})")
    return timings


def download_files(
    *,
    start_date,
    end_date,
    aoi_geometry,
    s3_sensor,
    cdse_credentials,
):
    # Transform parameters
    start_date = datetime.strptime(start_date, "%Y-%m-%d")
    end_date = datetime.strptime(end_date, "%Y-%m-%d")
    if s3_sensor == "SYN":
        instrument = "SYNERGY"
    else:
        instrument = s3_sensor

    # Download the data from CDSE
    download_from_cdse(
        start_date,
        end_date,
        aoi_geometry,
        s2_download_dir,
        s3_download_dir,
        cdse_credentials,
    )
    return instrument


def s2_preprocessing(*, s2_bands, ratio):
    # Sentinel-2 pre-processing
    s2.extract_mask_s2_bands(
        s2_download_dir,
        s2_processed_dir,
        bands=s2_bands,
    )
    s2.distance_to_clouds(
        s2_processed_dir,
        ratio=ratio,
    )
    footprint = s2.get_wkt_footprint(
        s2_processed_dir,
    )
    return footprint


def s3_preprocessing(
    *,
    footprint,
    s3_bands,
    instrument,
    snap_gpt_path,
    mosaic_days,
    step,
    use_snap_binning: bool = False,
):
    logger = logging.getLogger(__file__)
    s3_timings: list[Tuple[str, float]] = []
    # Sentinel-3 pre-processing
    with Timer(dsc="S3 binning", logger=logger, add_to=s3_timings):
        if use_snap_binning:
            logger.info("Using SNAP binning")
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
        else:
            logger.info("Using built-in binning implementation")
            binning.binning_s3_py(
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

    with Timer(dsc="S3 median composite", logger=logger, add_to=s3_timings):
        s3.produce_median_composite(
            s3_binning_dir,
            s3_composites_dir,
            mosaic_days=mosaic_days,
            step=step,
            s3_bands=None,
        )

    with Timer(dsc="S3 smoothing", logger=logger, add_to=s3_timings):
        s3.smoothing(
            s3_composites_dir,
            s3_blured_dir,
            std=1,
            preserve_nan=False,
        )

    with Timer(dsc="S3 reformat", logger=logger, add_to=s3_timings):
        s3.reformat_s3(
            s3_blured_dir,
            s3_calibrated_dir,
        )

    with Timer(dsc="S3 reproject and crop", logger=logger, add_to=s3_timings):
        s3.reproject_and_crop_s3(
            s3_calibrated_dir,
            s2_processed_dir,
            s3_reprojected_dir,
        )

    for desc, t in s3_timings: 
        logger.info(f"[Timings] {t} ({desc})")


def perform_efast(*, start_date, end_date, step, ratio):
    # Perform EFAST fusion
    for date in rrule.rrule(
        rrule.DAILY,
        dtstart=start_date + timedelta(step),
        until=end_date - timedelta(step),
        interval=step,
    ):
        logger = logging.getLogger(__name__)
        logger.info(f"Performing efast for {date}")
        efast.fusion(
            date,
            s3_reprojected_dir,
            s2_processed_dir,
            fusion_dir,
            product="REFL",
            ratio=ratio,
            max_days=100,
            minimum_acquisition_importance=0,
        )


def download_from_cdse(
    start_date, end_date, aoi_geometry, s2_download_dir, s3_download_dir, credentials
):

    # First download Sentinel-3 SYN data
    results = query.query(
        "Sentinel3",
        start_date=start_date,
        end_date=end_date,
        geometry=aoi_geometry,
        instrument="SYNERGY",
        productType="SY_2_SYN___",
        timeliness="NT",
    )
    download_list_safe(
        [result["id"] for result in results.values()],
        outdir=s3_download_dir,
        threads=1,
        **credentials,
    )
    for zip_file in s3_download_dir.glob("*.zip"):
        print(zip_file)
        with zipfile.ZipFile(zip_file, "r") as zip_ref:
            zip_ref.extractall(s3_download_dir)

    # Then download Sentinel-2 L2A data
    results = query.query(
        "Sentinel2",
        start_date=start_date,
        end_date=end_date,
        geometry=aoi_geometry,
        productType="L2A",
    )
    download_list_safe(
        [result["id"] for result in results.values()],
        outdir=s2_download_dir,
        threads=3,
        **credentials,
    )
    for zip_file in s2_download_dir.glob("*.zip"):
        with zipfile.ZipFile(zip_file, "r") as zip_ref:
            zip_ref.extractall(s2_download_dir)

def download_list_safe(uids, username, password, outdir, threads=1, show_progress=True):
    from creodias_finder.download import _get_token, download

    if show_progress:
        pbar = tqdm(total=len(uids), unit="files")

    def _download(uid):
        token = _get_token(username, password)
        outfile = Path(outdir) / f"{uid}.zip"
        download(
            uid, username, password, outfile=outfile, show_progress=False, token=token
        )
        if show_progress:
            pbar.update(1)
        return uid, outfile

    paths = [_download(u) for u in uids]
    return paths


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-date", default="2022-06-01")
    parser.add_argument("--end-date", default="2022-06-30")
    parser.add_argument(
        "--aoi-geometry", default="POINT (-15.432283 15.402828)"
    )  # Dahra EC tower
    parser.add_argument("--s3-sensor", default="SYN")
    parser.add_argument(
        "--s3-bands", nargs="+", default=["SDR_Oa04", "SDR_Oa06", "SDR_Oa08", "SDR_Oa17"]
    )
    parser.add_argument("--s2-bands", nargs="+", default=["B02", "B03", "B04", "B8A"])
    parser.add_argument("--mosaic-days", type=int, default=100)
    parser.add_argument("--step", required=False, default=2, type=int)
    parser.add_argument("--cdse-credentials", default=CREDENTIALS)
    parser.add_argument("--snap-gpt-path", required=False, default="gpt")
    parser.add_argument("--ratio", required=False, type=int, default=30)
    parser.add_argument(
        "--use-snap-binning",
        action="store_true",
        help="Use SNAP for binning rather than built-in Python implementation"
    )
    #parser.add_argument("--sigma-doy", type=float, default=20)

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
        snap_gpt_path=args.snap_gpt_path,
        ratio=args.ratio,
        use_snap_binning=args.use_snap_binning,
        #sigma=args.sigma_doy,
    )
