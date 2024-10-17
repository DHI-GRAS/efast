from scipy.ndimage import distance_transform_edt
import numpy as np
import xarray as xr
from openeo.udf import XarrayDataCube

def apply_datacube(cube: XarrayDataCube, context: dict) -> XarrayDataCube:
    array = cube.get_array()
    distance = distance_transform_edt(array)
    clipped = np.clip(distance, 0, 255)
    return XarrayDataCube(xr.DataArray(clipped, dims=["bands", "t", "y", "x"]))
