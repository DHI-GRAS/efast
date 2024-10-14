# #from scipy.ndimage import distance_transform_edt
# #import numpy as np
# import xarray as xr
from openeo.udf import XarrayDataCube

def apply_datacube(cube: XarrayDataCube, context: dict) -> XarrayDataCube:
    #distance = distance_transform_edt(cube)
    #return np.clip(distance, 0, 255)
    return cube
