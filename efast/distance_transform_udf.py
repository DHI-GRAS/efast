from openeo.udf.debug import inspect
from scipy.ndimage import distance_transform_edt
import numpy as np
import xarray as xr
from openeo.udf import XarrayDataCube

def apply_datacube(cube: XarrayDataCube, context: dict) -> XarrayDataCube:
    array = cube.get_array()
    inspect(array.dims, message="array dims")
    inspect(array.sizes, message="array sizes")
    distance = distance_transform_edt(array)
    clipped = np.clip(distance, 0, 255)
    return XarrayDataCube(xr.DataArray(clipped, dims=["t", "bands", "y", "x"]))
