from openeo import UDF
from openeo.udf import XarrayDataCube
from openeo.udf.run_code import execute_local_udf
import xarray as xr
import pandas as pd
import numpy as np

from efast.openeo.udf.compute_s3_composite_weights import apply_datacube


DATE_FORMAT = "%Y-%m-%d"


def create_dummy_data():
    time = pd.date_range("2018-06-01", periods=30, freq="D")
    x = np.arange(7)
    y = np.arange(6)
    bands = ["red", "green", "blue"]
    data = np.random.rand(len(time), len(x), len(y), len(bands))

    ds = xr.DataArray(
            data,
            dims=["t", "x", "y", "bands"],
            coords={
                "t": time,
                "x": x,
                "y": y,
                "bands": bands,
                }
            )
    return ds


def create_dummy_dtc():
    time = pd.date_range("2018-06-01", periods=30, freq="D")
    x = np.arange(7)
    y = np.arange(6)
    bands = ["distance_to_cloud"]
    #data = np.random.rand(len(time), len(x), len(y), len(bands))
    data = np.random.rand(len(time), len(bands), len(y), len(x))

    da = xr.DataArray(
            data,
            #dims=["t", "x", "y", "bands"],
            dims=["t", "bands", "y", "x"],
            coords={
                "t": time,
                "x": x,
                "y": y,
                "bands": bands,
                }
            )
    return da


def test_time_difference_score():
    data = create_dummy_dtc()
    udf = UDF.from_file("efast/openeo/udf/compute_s3_composite_weights.py")
    target_date = "2018-06-18"
    context = {
        "target_date": target_date,
        "sigma_doy": 10,
        }
    apply_datacube(XarrayDataCube(data), context)
    res = execute_local_udf(udf, data, fmt="netcdf")

    return True



