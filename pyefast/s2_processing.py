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

import re
import xml.etree.ElementTree as ET

import numpy as np
import pyproj
import rasterio
import scipy as sp

from shapely.geometry import box
from shapely.ops import transform
from tqdm import tqdm

# Mapping of Sentinel-2 bands names to bands ids
BANDS_IDS = {
    "B02": "1",
    "B03": "2",
    "B04": "3",
    "B05": "4",
    "B06": "5",
    "B07": "6",
    "B08": "7",
    "B8A": "8",
    "B11": "11",
    "B12": "12",
}


def extract_mask_s2_bands(
    input_dir, output_dir, bands=["B02", "B03", "B04", "B8A"], resolution=20
):
    """
    Extract specified Sentinel-2 bands from .SAFE file, mask clouds and shadows using the SLC mask
    and save to multi-band GeoTIFF file.

    Parameters
    ----------
    input_dir : pathlib.Path
        The directory where the Sentinel-2 .SAFE images are stored.
    output_dir: pathlib.Path
        The directory where the Sentinel-2 GeoTIFF images are to be stored.
    bands: list [str], optional
        List of bands names to be extracted from the .SAFE file.
        Defaults to ["B02", "B03", "B04", "B08"]
    resolution: int, optional
        Spatial resolution of the bands to be extracted.
        Defaults to 20.

    Returns
    -------
    None
    """
    for p in input_dir.iterdir():
        band_paths = [
            list(p.glob(f"GRANULE/*/IMG_DATA/R{resolution}m/*{band}*.jp2"))[0]
            for band in bands
        ]

        # Find S2 BOA offsets
        tree = ET.parse(p / "MTD_MSIL2A.xml")
        root = tree.getroot()
        offset_list = root.findall(".//BOA_ADD_OFFSET")
        offsets = {el.attrib["band_id"]: el.text for el in offset_list}

        # Extract rasterio profile
        with rasterio.open(band_paths[0]) as src:
            profile = src.profile.copy()

        # Read SLC cloud mask
        mask_path = list(p.glob(f"GRANULE/*/IMG_DATA/R{resolution}m/*SCL*"))[0]
        with rasterio.open(mask_path) as mask_src:
            mask = mask_src.read(1)
        mask = (mask == 0) | (mask == 3) | (mask > 7)

        # Combine bands and mask
        s2_image = np.zeros(
            (len(bands), profile["height"], profile["width"]), "float32"
        )
        for i, band_path in enumerate(band_paths):
            band = bands[i]
            band_id = BANDS_IDS.get(band)
            offset = int(offsets.get(band_id, 0))
            with rasterio.open(band_path) as src:
                raw_data = src.read(1).astype("int16")
                data = (raw_data - offset) / 10000
                data[data < 0] = 0
                data[mask] = 0
                s2_image[i] = data

        # Save file
        profile.update(
            {"driver": "GTiff", "count": len(bands), "dtype": "float32", "nodata": 0}
        )
        out_path = output_dir / f"{str(p.name).rstrip('.SAFE')}_REFL.tif"
        with rasterio.open(out_path, "w", **profile) as dst:
            dst.write(s2_image)


def distance_to_clouds(dir_s2, ratio=30, tolerance_percentage=0.05):
    """
    Calculate distance to nearest cloud (or other no-data part of the image) and save it at
    roughly OLCI spatial resolution (300 m)

    Parameters
    ----------
    dir_s2 : pathlib.Path
        The directory where the Sentinel-2 images are stored. Clouds and shadows should the masked
        using 0.
    ratio: int, optional
        The (rough) ratio between resolution of Sentinel-2 and Sentinel-3 images. Defaults to 30.
    tolerance_percentage: float, optional
        Fraction of low-resolution (Sentinel-3) pixel which can be covered by Sentinel-2 resolution
        cloudy pixels before the low-resolution pixel is considered to be cloudy. Defaults to 0.05.

    Returns
    -------
    None

    References
    ----------
    ..  [Senty2024] Senty, P., Guzinski, R., Grogan, K., Buitenwerf, R., Ardö, J., Eklundh, L.,
        Koukos, A., Tagesson, T., and Munk, M. (2024). Fast Fusion of Sentinel-2 and Sentinel-3
        Time Series over Rangelands. Remote Sensing 16, 1833. https://doi.org/10.3390/rs16111833
    ..  [Griffiths2013] Griffiths, P.; van der Linden, S.; Kuemmerle, T.; Hostert, P. A Pixel-Based
        Landsat Compositing Algorithm for Large Area Land Cover Mapping. IEEE J. Sel. Top. Appl.
        Earth Obs. Remote Sens. 2013, 6, 2088–2101. https://doi.org/10.1109/JSTARS.2012.2228167.
    """

    sen2_paths = dir_s2.glob("*REFL.tif")

    for sen2_path in tqdm(sen2_paths):
        # Read s2 image
        with rasterio.open(sen2_path) as src:
            s2_hr = src.read(1)
            s2_profile = src.profile

        # Check if a Sentinel-3 pixel is complete
        s2_block = (
            (s2_hr == 0)
            .reshape(s2_hr.shape[0] // ratio, ratio, s2_hr.shape[1] // ratio, ratio)
            .mean(3)
            .mean(1)
        )

        # Distance to cloud score
        mask = s2_block < tolerance_percentage
        distance_to_cloud = sp.ndimage.distance_transform_edt(mask)
        distance_to_cloud = np.clip(distance_to_cloud, 0, 255)

        # Update transform
        s2_resolution = (s2_profile["transform"] * (1, 0))[0] - (
            s2_profile["transform"] * (0, 0)
        )[0]
        longitude_origin, latitude_origin = s2_profile["transform"] * (0, 0)
        lr_transform = rasterio.Affine(
            ratio * s2_resolution,
            0,
            longitude_origin,
            0,
            -ratio * s2_resolution,
            latitude_origin,
        )

        # Update profile to sentinel-3 geometry
        s2_profile.update(
            {
                "width": mask.shape[1],
                "height": mask.shape[0],
                "transform": lr_transform,
            }
        )

        # Update profile with a new dtype
        s2_profile.update({"count": 1})

        # Save output
        out_path = re.sub("_[A-Z]*\.tif", "_DIST_CLOUD.tif", str(sen2_path))
        with rasterio.open(out_path, "w", **s2_profile) as dst:
            dst.write(distance_to_cloud[np.newaxis])


def get_wkt_footprint(dir_s2, crs="EPSG:4326"):
    """
    Get the footprint (bounds) of the first image in the directory in WKT format

    Parameters
    ----------
    dir_s2 : pathlib.Path
        The directory where the Sentinel-2 images are stored.
    crs: str, optional
        The projection of the returned footrpint. Defaults to EPSG:4326

    Returns
    -------
    footprint: str
        The footprint in WKT format
    """

    image_path = list(dir_s2.glob("*REFL.tif"))[0]

    # Get images's bounds and CRS
    with rasterio.open(image_path) as src:
        bounds = src.bounds
        image_crs = src.crs

    # Ensure footprint is in desired CRS
    polygon = box(*bounds)
    if image_crs != crs:
        transformer = pyproj.Transformer.from_proj(
            pyproj.Proj(image_crs), pyproj.Proj(crs), always_xy=True
        )
        polygon = transform(transformer.transform, polygon)

    # Step 4: Convert to WKT
    footprint = polygon.wkt

    return footprint
