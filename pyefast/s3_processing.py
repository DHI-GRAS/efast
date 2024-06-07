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

import os
import re
from datetime import datetime

import astropy.convolution as ap
import numpy as np
import pandas as pd
import rasterio
import scipy as sp
from dateutil import rrule
from rasterio import shutil as rio_shutil
from rasterio.enums import Resampling
from rasterio.vrt import WarpedVRT
from snap_graph.snap_graph import SnapGraph
from tqdm import tqdm


def binning_s3(
    download_dir,
    binning_dir,
    footprint,
    s3_bands=["SDR_Oa04", "SDR_Oa06", "SDR_Oa08", "SDR_Oa17"],
    instrument="OL",
    max_zenith_angle=30,
    crs="EPSG:4326",
    aggregator="mean",
    snap_gpt_path="gpt",
    snap_memory="8G",
    snap_parallelization=1,
):
    """
    Create single-band composites of Sentinel-3 data
    Parameters
    ----------
    download_dir : Path
        folder where the Sentinel-3 files were downloaded
    binning_dir : Path
        folder where the Sentinel-3 composites will be stored
    s3_bands : list, optional.
        Sentinel-3 variables to consider (e.g. ['FAPAR']).
        Defaults to '["SDR_Oa04", "SDR_Oa06", "SDR_Oa08", "SDR_Oa17"]'
    instrument : str
        specify if 'OL' or 'SYNERGY' is used, it affects the choice of masks
    max_zenith_angle : int
        all the acquisitions with a higher viewing zenith angle than this value will be discarded
    crs : str
        coordinate reference system to reproject on (default: "EPSG:4326")
    footprint : geometry
         Area of interest as well-known text string, the Sentinel-3 acquisitions will be cropped to
         this area
    aggregator: str
        'mean' or 'median'
    snap_gpt_path : str
        path to SNAP's gpt.exe
    snap_memory : str
        memory to allocate to SNAP (default: '8G')
    snap_parallelization : int
        parallelization of SNAP processing

    Returns
    -------
    None
    """

    # sort files by dates
    if s3_bands is None:
        s3_bands = ["FAPAR"]
    sen3_paths = list(download_dir.glob("*.SEN3"))
    date_strings = pd.to_datetime(
        [os.path.split(sen3_path)[-1].split("____")[1][:8] for sen3_path in sen3_paths]
    )
    sen3_paths = [element for _, element in sorted(zip(date_strings, sen3_paths))]

    for i, sen3_path in enumerate(sen3_paths):
        output_path = os.path.join(
            binning_dir,
            sen3_path.stem + ".tif",
        )

        variables = s3_bands.copy()
        # Special case - FAPAR name changed from OGVI to GIFAPAR
        for i, variable in enumerate(s3_bands):
            if variable == "FAPAR":
                if "ogvi.nc" in os.listdir(sen3_path):
                    variables[i] = "OGVI"
                elif "gifapar.nc" in os.listdir(sen3_path):
                    variables[i] = "GIFAPAR"
                else:
                    continue

        # For OL products, which acquisitions to consider - not cloudy, over land and with a low
        # angle
        binning_variable_list = [
            {
                "name": f"{variable}_filtered",
                "expr": variable,
                "validExpr": f"OZA<{max_zenith_angle} and LQSF_CLOUD<254 and LQSF_CLOUD_MARGIN<254"
                + f" and LQSF_CLOUD_AMBIGUOUS<254 and LQSF_{variable}_FAIL<254 and LQSF_LAND>254",
            }
            for variable in variables
        ]
        if instrument != "OL":
            # For SYN products, which acquisitions to consider - not cloudy, over land and with a
            # low angle
            binning_variable_list = [
                {
                    "name": f"{variable}_filtered",
                    "expr": variable,
                    "validExpr": "CLOUD_flags_CLOUD<254 and "
                    "CLOUD_flags_CLOUD_AMBIGUOUS<254 and CLOUD_flags_CLOUD_MARGIN<254 and "
                    "SYN_flags_SYN_land>254",
                }
                for variable in variables
            ]
        binning_aggregator_list = [
            {
                "type": "AVG",
                "varName": f"{variable}_filtered",
                "targetName": f"{variable}",
            }
            for variable in variables
        ]
        if aggregator == "median":
            binning_aggregator_list = [
                {
                    "type": "PERCENTILE",
                    "varName": f"{variable}_filtered",
                    "targetName": f"{variable}",
                    "percentage": "50",
                }
                for variable in variables
            ]

        graph = SnapGraph()
        input_node_ids = []
        read_node_id = graph.read_op(str(sen3_path))  # read
        input_node_ids.append(graph.reproject_op(read_node_id, crs))  # reproject
        binning_node_id = graph.binning_op(
            source_product_list=input_node_ids,
            aggregator_list=binning_aggregator_list,
            variable_list=binning_variable_list,
            spatial_resolution=0.3,
            super_sampling=2,
            region=footprint,
            output_file=output_path,
        )
        string = ""
        for variable in variables:
            if aggregator == "median":
                string += f"{variable}_p50,"
            else:
                string += f"{variable}_mean,"
        subset_node_id = graph.subset_op(binning_node_id, string[:-1])
        graph.write_op(subset_node_id, output_path)  # export as .tif
        graph.run(snap_gpt_path, snap_parallelization, snap_memory)


def produce_median_composite(
    dir_s3, composite_dir, step=5, mosaic_days=100, s3_bands=None, D=20, sigma_doy=10
):
    """
    Create weighted composites of Sentinel-3 images.

    Parameters
    ----------
    dir_s3: pathlib.Path
        The directory containing the Sentinel-3 images.
    composite_dir: pathlib.Path
        The directory where the composite images will be saved.
    step: int, optional
        The number of days between each composite image, default is 5.
    mosaic_days: int, optional
        The number of days for which images are combined, default is 100.
    s3_bands: list, optional
        List of bands to include in the composite. If not provided, all bands will be used.
    D: int, optional
        The distance (in pixels) beyond which a cloud has no impact on the score, default is 20.
    sigma_doy: int, optional
        The standard deviation of the weight assigned to each image based on its distance to the
        central date, default is 10.

    Returns
    -------
    None
    """
    sen3_paths = list(dir_s3.glob("S3*.tif"))
    s3_dates = pd.to_datetime(
        [
            re.match(".*__(\d{8})T.*\.tif", sen3_path.name).group(1)
            for sen3_path in sen3_paths
        ]
    )
    sen3_paths = np.array(
        [sen3_path for _, sen3_path in sorted(zip(s3_dates, sen3_paths))]
    )
    s3_dates = pd.to_datetime(sorted(s3_dates))

    target_dates = rrule.rrule(
        rrule.DAILY,
        dtstart=s3_dates[0],
        until=s3_dates[-1],
        interval=step,
    )
    for middle_date in tqdm(target_dates):
        indices = list(
            np.argwhere(
                np.abs((s3_dates - middle_date).days) <= mosaic_days / 2
            ).flatten()
        )

        if len(indices) == 0:
            continue

        composites_paths = sen3_paths[indices]

        output_path = os.path.join(
            composite_dir,
            f'composite_{datetime.strftime(middle_date, "%Y-%m-%d")}.tif',
        )

        s3_stack = []
        for file in composites_paths:
            with rasterio.open(file) as src:
                profile = src.profile
                if s3_bands:
                    s3_image = src.read(s3_bands)
                    s3_image[
                        :, np.logical_not(np.abs(np.mean(s3_image, axis=0)) < 5)
                    ] = np.nan  # remove abnormally high values
                    s3_stack.append(s3_image)
                    profile.update({"count": len(s3_bands)})
                else:
                    s3_stack.append(src.read())
        s3_stack = np.array(s3_stack)

        # compute distance of each pixel to the closest cloud, and the corresponding score
        distance_to_cloud = []
        for i in range(s3_stack.shape[0]):
            mask = np.logical_not(np.isnan(s3_stack[i, 0]))
            mask[:2, :], mask[-2:, :], mask[:, :2], mask[:, -2:] = 1, 1, 1, 1
            distance_to_cloud.append(sp.ndimage.distance_transform_edt(mask))
        distance_to_cloud = np.array(distance_to_cloud)
        distance_score = (distance_to_cloud - 1) / D
        distance_score[distance_score < 0] = 0
        distance_score[distance_score > 1] = 1

        # compute temporal weight, based on the distance to middle_date
        doy_score = np.exp(
            -1
            / 2
            * (np.array((s3_dates[indices] - middle_date).days) ** 2 / sigma_doy**2)
        )
        doy_score = doy_score / np.max(doy_score)

        # combined score
        score = distance_score * doy_score[:, np.newaxis, np.newaxis]

        # normalize score and compute weighted average
        score[np.isnan(s3_stack[:, 0])] = 0
        score = score / (np.sum(score, axis=0)[np.newaxis] + 1e-5)
        score = np.expand_dims(score, axis=1)
        weighted_composite = np.nansum(score * s3_stack, axis=0)
        weighted_composite[:, np.sum(score[:, 0], axis=0) == 0] = np.nan

        with rasterio.open(output_path, "w", **profile) as dst:
            dst.write(weighted_composite)


def smoothing(s3_dir, smoothed_dir, product="composite", std=1, preserve_nan=True):
    """
    Smooths Sentinel-3 images using a 2D Gaussian kernel.

    Parameters
    ----------
    s3_dir : Path
        The path to the directory containing the Sentinel-3 composites.
    smoothed_dir: Path
        The path to the directory where the smoothed images will be saved.
    product: str, optional
        Product name, which should be in the filename and is used to select files for smoothing.
        Defaults to "composite"
    std: int, optional
        The standard deviation of the Gaussian kernel used for smoothing. Defaults to 1.
    preserve_nan: bool, optional
        Indicates whether NaN values should be preserved in the smoothed images. Defaults to True.

    Returns
    -------
    None
    """
    s3_paths = np.array(list(s3_dir.glob(f"*{product}*.tif")))

    for s3_path in tqdm(s3_paths, "Smoothing"):
        with rasterio.open(s3_path) as src:
            s3 = src.read()
            profile = src.profile
        smoothed_s3 = s3.copy()
        kernel = ap.Gaussian2DKernel(
            x_stddev=std, y_stddev=std, x_size=4 * std + 1, y_size=4 * std + 1
        )
        for band in range(s3.shape[0]):
            smoothed_s3[band] = ap.convolve(
                s3[band], kernel, boundary="extend", preserve_nan=preserve_nan
            )
        with rasterio.open(
            os.path.join(smoothed_dir, os.path.split(s3_path)[-1]), "w", **profile
        ) as dst:
            dst.write(smoothed_s3)


def reformat_s3(
    s3_dir,
    calibrated_s3_dir,
    product="composite",
    scaling_factor=1,
):
    """Reformat Sentinel-3 images to a format suitable for analysis.

    Parameters
    ----------
    s3_dir : Path
        Folder containing the Sentinel-3 files to be reformatted.
    calibrated_s3_dir : Path
        Folder where the reformatted Sentinel-3 files will be stored.
    product: str, optional
        Product name, which should be in the filename and is used to select files for reformatting.
        Defaults to "composite".
    scaling_factor : int, optional
        Factor by which to scale the Sentinel-3 images, default is 1.

    Returns
    -------
    None
    """

    # Sentinel-3 paths or dates
    s3_paths = np.array(list(s3_dir.glob(f"*{product}*.tif")))

    # Use these coefficients to calibrate Sentinel-3 on Sentinel-2 values
    for s3_path in s3_paths:
        with rasterio.open(s3_path) as src:
            s3_image = src.read()
            s3_profile = src.profile

        for band in range(s3_image.shape[0]):
            s3_image[band] = scaling_factor * s3_image[band]
        s3_image[s3_image < 0] = 0  # avoid negative values
        s3_image = s3_image.astype("float32")

        # update profile
        s3_profile.update({"count": s3_image.shape[0], "dtype": "float32"})

        # export image
        export_path = os.path.join(calibrated_s3_dir, os.path.split(s3_path)[-1])
        with rasterio.open(export_path, "w", **s3_profile) as dst:
            dst.write(s3_image)


def reproject_and_crop_s3(s3_dir, s2_dir, export_dir):
    """
    Create single-band composites of Sentinel-3 data

    Parameters
    ----------
    s3_dir : Path
        folder where the Sentinel-3 composites are stored
    s2_dir : Path
        folder where the Sentinel-2 images were downloaded
    export_dir : Path
        folder where the Sentinel-3 composites will be reprojected to Sentinel-2 CRS

    Returns
    -------
    None
    """

    sen3_paths = list(s3_dir.glob("*.tif"))
    sen2_paths = list(s2_dir.glob("*CLOUD.tif"))
    if len(sen2_paths) == 0:
        sen2_paths = list(s2_dir.rglob("*CLOUD.tif"))

    with rasterio.open(sen2_paths[0]) as src:
        lr_s2_profile = src.profile

    for sen3_path in tqdm(sen3_paths, "Sentinel-3 reprojection"):
        # create a dictionary containing the desired transform, height, width, and crs.
        vrt_options = {
            "transform": lr_s2_profile["transform"],
            "height": lr_s2_profile["height"],
            "width": lr_s2_profile["width"],
            "crs": lr_s2_profile["crs"],
            "resampling": Resampling.cubic,
        }

        # Read the JRC data as client, using the master properties
        with rasterio.open(sen3_path) as client_src:
            with WarpedVRT(client_src, **vrt_options) as vrt:
                # At this point 'vrt' is a full dataset with dimensions,
                # CRS, and spatial extent matching 'vrt_options'.

                # Export
                _, name = os.path.split(sen3_path)
                outfile = os.path.join(export_dir, name)
                rio_shutil.copy(vrt, outfile, driver="GTiff")
