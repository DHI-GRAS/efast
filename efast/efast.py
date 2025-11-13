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

import numpy as np
import pandas as pd
import rasterio
import rasterio.windows
import scipy.ndimage

from scipy.interpolate import interp1d
from tqdm import tqdm


def fusion(
    pred_date,
    lr_dir,
    hr_dir,
    fusion_dir,
    product,
    max_days=30,
    sigma=20,
    ratio=30,
    date_position=2,
    D=15,
    add_overview=False,
    extent=None,
    minimum_acquisition_importance=2,
):
    """
    A function that combines time-series of low-resolution and high-resolution images to produce a
    synthetic high-resolution image for the prediction date.

    Parameters
    ----------
    pred_date : datetime.datetime
        The date of the predicted image.
    lr_dir : pathlib.Path
        The directory where the low-resolution images are stored.
    hr_dir : pathlib.Path
        The directory where the high-resolution images are stored.
    fusion_dir : pathlib.Path
        The directory where the fused images will be stored.
    product: string
        Product name, which should be in the filename of high-resolution images and is used to
        select files for fusion
    max_days : int, optional
        The maximum number of days allowed between the acquisition dates and the predicted date
        by default 30.
    sigma : int, optional
        The standard deviation for the temporal weights, by default 20 days.
    ratio : int, optional
        The ratio between the resolution of the high- and low-resolution images, by default 30.
    date_position : int, optional
        The position of the date in the filename of the high-resolution images, by default 2.
    D : int, optional
        The distance (in coarse pixels) beyond which a cloud has no impact on the score, by default
        15.
    add_overview : bool, optional
        If True, an overview image will be added, by default True.
    extent : tuple, optional
        The extent of the image to be used, by default the whole high-resolution image is used.
    minimum_acquisition_importance : float, optional
        The minimum acquisition importance to be used in the averaging process, by default 2.

    Returns
    -------
    None

    References
    ----------
    ..  [Senty2024] Senty, P., Guzinski, R., Grogan, K., Buitenwerf, R., Ardö, J., Eklundh, L.,
        Koukos, A., Tagesson, T., and Munk, M. (2024). Fast Fusion of Sentinel-2 and Sentinel-3
        Time Series over Rangelands. Remote Sensing 16, 1833. https://doi.org/10.3390/rs16111833
    """

    hr_suffix = product

    # High-resolution images
    hr_paths_c = sorted(hr_dir.glob("*DIST_CLOUD.tif"))
    hr_paths_hr = sorted(hr_dir.glob(f"*{hr_suffix}.tif"))
    hr_dates = pd.to_datetime(
        [os.path.split(hr_path)[-1].split("_")[date_position] for hr_path in hr_paths_c]
    )
    hr_paths_c = np.array([hr_path for _, hr_path in sorted(zip(hr_dates, hr_paths_c))])
    hr_paths_hr = np.array(
        [hr_path for _, hr_path in sorted(zip(hr_dates, hr_paths_hr))]
    )
    hr_dates = pd.to_datetime(sorted(hr_dates))

    # Low-resolution images
    lr_paths = sorted(lr_dir.glob("composite*.tif"))
    lr_dates = pd.to_datetime(
        [os.path.split(lr_path)[-1].split("_")[1][:-4] for lr_path in lr_paths]
    )
    lr_paths = np.array([lr_path for _, lr_path in sorted(zip(lr_dates, lr_paths))])
    lr_dates = pd.to_datetime(sorted(lr_dates))

    # Images used in the averaging process
    lr_potential_dates = np.array(
        [
            -1.5 * max_days < (lr_date - pred_date).days < 1.5 * max_days
            for lr_date in lr_dates
        ]
    )
    hr_potential_dates = np.array(
        [-max_days < (hr_date - pred_date).days < max_days for hr_date in hr_dates]
    )
    lr_paths = lr_paths[lr_potential_dates]
    hr_paths_c = hr_paths_c[hr_potential_dates]
    hr_paths_hr = hr_paths_hr[hr_potential_dates]

    # Convert dates to numbers
    t_s3 = np.array((lr_dates[lr_potential_dates] - pred_date).days)
    t_s2 = np.array((hr_dates[hr_potential_dates] - pred_date).days)

    # Crop extent
    lr_window = None
    hr_window = None
    if extent is not None:
        x_min, x_max, y_min, y_max = extent
        lr_window = rasterio.windows.Window.from_slices((x_min, x_max), (y_min, y_max))
        hr_window = rasterio.windows.Window.from_slices(
            (ratio * x_min, ratio * x_max), (ratio * y_min, ratio * y_max)
        )

    # --- Section 2.3 from [Senty2024] (last paragraph) ---

    # Read low-resolution image
    if len(lr_paths) == 0:
        return
    with rasterio.open(lr_paths[0]) as src:
        lr_profile = src.profile
        bands = src.count
    if lr_window is not None:
        lr_profile.update(
            {
                "width": lr_window.width,
                "height": lr_window.height,
                "transform": rasterio.windows.transform(
                    lr_window, lr_profile["transform"]
                ),
            }
        )

    lr_values = np.zeros((len(t_s3), bands, lr_profile["height"], lr_profile["width"]))
    for i, lr_path in enumerate(lr_paths):
        with rasterio.open(lr_path) as src:
            lr_values[i, :, :, :] = src.read(window=lr_window)

    # Prepare low-resolution rasters
    lr_i = np.nan * np.zeros(
        (len(t_s2), lr_values.shape[1], lr_profile["height"], lr_profile["width"]),
        lr_profile["dtype"],
    )
    lr_p = np.nan * np.zeros(
        (lr_values.shape[1], lr_profile["height"], lr_profile["width"]),
        lr_profile["dtype"],
    )

    # Interpolate low-resolution values temporally
    for row in tqdm(range(lr_profile["height"]), "Low-resolution image interpolation"):
        for col in range(lr_profile["width"]):
            for band in range(lr_values.shape[1]):
                # list of available low-resolution pixels
                y = lr_values[:, band, row, col]
                train_dates = t_s3[np.logical_not(np.isnan(y))]
                y = y[np.logical_not(np.isnan(y))]
                pred_dates = np.concatenate(([0], t_s2))
                if len(train_dates) > 3:
                    cs = interp1d(train_dates, y, fill_value="extrapolate")
                    interpolations = cs(pred_dates)
                    lr_p[band, row, col] = interpolations[0]
                    lr_i[:, band, row, col] = interpolations[1:]

    # --- Section 2.5 from [Senty2024] ---

    # Read distance-to-cloud scores for high-resolution images
    distance_to_cloud = np.zeros((len(t_s2), lr_profile["height"], lr_profile["width"]))
    for i, hr_path_c in enumerate(hr_paths_c):
        with rasterio.open(hr_path_c) as src:
            distance_to_cloud[i] = src.read(1, window=lr_window)

    # Compute the weights of each high-resolution image for every low-resolution pixel, temporal
    # times optical Gaussian (t)
    wt_i = np.exp(-((t_s2 / sigma) ** 2))
    # Ramp (distance) - 0 for 1 km away from a cloud, 1 for 10 km
    wo_i = (distance_to_cloud - 1) / D
    wo_i[wo_i < 0] = np.nan
    wo_i[wo_i > 1] = 1
    # Product
    w_i = wo_i * wt_i.reshape((len(t_s2), 1, 1))

    # Normalize the weights
    Sw_i = np.nansum(w_i, axis=0)
    Sw_i[Sw_i == 0] = np.nan
    w_i = w_i / Sw_i[np.newaxis]
    w_i = w_i.reshape((len(t_s2), 1, lr_profile["height"], lr_profile["width"]))

    # Select the best quality high-resolution images
    acquisition_importance = np.nansum(w_i, axis=(1, 2, 3)) / np.prod(w_i.shape[1:])
    hr_selected = acquisition_importance > minimum_acquisition_importance / len(t_s2)
    w_is = w_i[hr_selected]
    n_selected = np.sum(hr_selected)

    # Initialize the mosaic rasters
    hr_m = np.zeros(
        (lr_values.shape[1], lr_profile["height"] * ratio, lr_profile["width"] * ratio),
        lr_profile["dtype"],
    )
    lr_m = np.zeros(
        (lr_values.shape[1], lr_profile["height"] * ratio, lr_profile["width"] * ratio),
        lr_profile["dtype"],
    )
    wu = np.zeros((1, lr_profile["height"] * ratio, lr_profile["width"] * ratio))

    # Fill the mosaics
    for i in tqdm(range(n_selected), "High-resolution mosaic"):
        # Bilinear interpolation of weights
        w_is[i, np.isnan(w_is[i])] = 0
        wu_i = upsample_array(w_is[i], ratio=ratio, order=1)

        # Read high-resolution image and multiply by the weight
        with rasterio.open(hr_paths_hr[hr_selected][i]) as src:
            hr_i = src.read(window=hr_window)

        hr_i[np.isnan(hr_i)] = 0
        wu_i[0, np.logical_or(np.isnan(hr_i[0]), hr_i[0] == 0)] = 0
        hr_m += hr_i * wu_i

        # Apply the same weights to low-resolution image
        lr_m += upsample_array(lr_i[hr_selected][i], ratio=ratio) * wu_i

        # Counter of the sum of weights (wu is the normalization coefficient)
        wu += wu_i

    # Normalize the mosaics
    wu[wu == 0] = np.nan
    hr_m = hr_m / wu
    lr_m = lr_m / wu

    # --- Section 2.4 from [Senty2024] ---

    # Fusion, eq. (2) from [Senty2024]
    lr_p = upsample_array(lr_p, ratio=ratio)
    fusion = lr_p + hr_m - lr_m

    # --- Save outputs ---

    with rasterio.open(hr_paths_hr[0]) as src:
        profile = src.profile

    # Export
    export_path = os.path.join(
        fusion_dir,
        f'{product}_{pred_date.strftime("%Y%m%d")}.tif',
    )

    if lr_window is not None:
        profile.update(
            {
                "width": hr_window.width,
                "height": hr_window.height,
                "transform": rasterio.windows.transform(
                    hr_window, profile["transform"]
                ),
            }
        )

    with rasterio.open(export_path, "w", **profile) as dst:
        dst.write(fusion)
        # TODO - change add_overview to save_as_cog
        if add_overview:
            dst.build_overviews([2, 4, 8, 16], rasterio.enums.Resampling.average)
            dst.update_tags(ns="rio_overview", resampling="average")


def upsample_array(C, fine_shape=None, ratio=15, kron=False, order=1):
    """
    Upsample an array `C` to a new shape using either np.kron or scipy.ndimage.zoom.

    Parameters
    ----------
    C: numpy.ndarray
        The input array to be up-sampled.
    fine_shape: tuple, optional
        The desired shape of the up-sampled array. If not specified, it is computed based on the
        shape of `C` and the `ratio`.
    ratio: int, optional
        The upsampling factor. Defaults to 15.
    kron: bool, optional
        Whether to use np.kron for upsampling. Defaults to False.
    order: int, optional
        The order of the spline interpolation to use when upsampling with scipy.ndimage.zoom.
        Defaults to 1.

    Returns
    -------
    numpy.ndarray
        The up-sampled array with the desired shape.
    """

    if fine_shape is None:
        fine_shape = (C.shape[0], ratio * C.shape[1], ratio * C.shape[2])

    # common height and width
    height = min(C.shape[1] * ratio, fine_shape[1])
    width = min(C.shape[2] * ratio, fine_shape[2])

    # upsample to high resolution
    F = np.zeros(fine_shape)
    for band in range(C.shape[0]):
        if kron:
            F[band, 0:height, 0:width] = np.kron(C[band], np.ones((ratio, ratio)))[
                0:height, 0:width
            ]
        else:
            F[band, 0:height, 0:width] = scipy.ndimage.zoom(
                C[band], ratio, order=order, grid_mode=True, mode="nearest"
            )[0:height, 0:width]

    return F
