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

See run_efast.py for an example using data located in test_data folder.

### Requirements
* [python](https://www.python.org/getit/)
* [esa-snap](https://step.esa.int/main/download/snap-download/) - needed for Sentinel-3 pre-processing only. Tested with version 9 and 10.

### Try it out

1. Clone the repository to your local machine.
2. Navigate to the root directory of the repository in your terminal.
3. [OPTIONAL but recommended] Create a virtual environment: `python3.<your python version> -m venv .venv`
3. Install the package: `pip install -e .`
4. Run the example: `python run_efast.py`

### Installation
Install the package using pip:

```bash
pip install git+https://github.com/DHI-GRAS/efast.git
```

### Usage
```python
import efast

...
efast.fusion(
    ...
)
```

### Develop
1. Clone the repository to your local machine.
2. Navigate to the root directory of the repository in your terminal.
3. [OPTIONAL but strongly recommended] Create a virtual environment: `python3.<your python version> -m venv .venv`
3. Install the package in dev mode: `pip install -e .[dev]`
