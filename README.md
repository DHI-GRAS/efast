# EFAST: Fusion of Sentinel-2 and Sentinel-3 Data

### Introduction

The EFAST package is designed for combining Sentinel-2 and Sentinel-3 data to produce frequent,
high-resolution images. This approach can also be applied to other satellite datasets,
such as Landsat and MODIS. The purpose of this package is to provide analysis-ready data,
that is, cloud-free optical images at regular time-steps.

### Motivation

Sentinel-3 satellites provide daily images of the same area with a coarse resolution of
about 300 meters on the ground. On the other hand, Sentinel-2 images have a higher resolution
of up to 10 meters but have a longer revisit time. By combining the two satellite datasets, it is
possible to obtain time-series of both high temporal and spatial resolution. This makes the EFAST
package a valuable tool for monitoring ecosystems and extracting key information.

### Use cases
The EFAST package is intended for users who:

1. Monitor ecosystems using Sentinel-2 or Landsat but are limited by the long revisit time of these satellites.
2. Want to make full use of the synergy between Sentinel-2, Sentinel-3, Landsat, and MODIS to obtain high-resolution time-series images.
3. Need cloud-free optical images at regular time-steps for their analysis.

### Reference
Senty, P., Guzinski, R., Grogan, K., Buitenwerf, R., Ardö, J., Eklundh, L., Koukos, A., Tagesson, T., and Munk, M. (2024).
Fast Fusion of Sentinel-2 and Sentinel-3 Time Series over Rangelands. Remote Sensing 16, 1833. [https://doi.org/10.3390/rs16111833](https://doi.org/10.3390/rs16111833).

### NDVI Example: Aarhus, Denmark in Spring 2021

![NDVI around Aarhus, Denmark in spring 2021][gif]

[gif]: images/ndvi.gif

The EFAST package can be used to generate cloud-free NDVI (Normalized Difference Vegetation Index) images,
as demonstrated by the example of Aarhus, Denmark in Spring 2021.

### How to Use EFAST

To use EFAST, follow these steps:

1. Open the fusion.py file.
2. Modify the path variable to specify your local path.
3. Enter your desired parameters, such as the Sentinel-2 tile number (tile_num), start date (start_date), and end date (end_date).

### Requirements

- setuptools
- numpy
- scipy
- tqdm
- scikit-learn
- rasterio
- pandas
- ipdb
- astropy
- python-dateutil
- snap-graph (available through a Git repository)

### Installation

1. Clone the repository to your local machine.
2. Navigate to the root directory of the repository in your terminal.
3. Run the following command to install the required packages: pip install -r requirements.txt
4. Run the following command to install the package: python setup.py install