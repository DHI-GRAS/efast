import numpy as np
import xarray as xr
from openeo.udf import XarrayDataCube
from openeo.udf.debug import inspect
from datetime import datetime

DATE_FORMAT = "%Y-%m-%d"
CONTEXT = {
    "target_date": "2018-06-18",
    "sigma_doy": 10,
    }

def apply_datacube(cube: XarrayDataCube, context: dict=CONTEXT) -> XarrayDataCube:
    distance_score = cube.get_array()
    inspect(distance_score["t"], level="INFO", message="inspecting time dimension")
    target_date = datetime.strptime(get_from_context(context, "target_date"), DATE_FORMAT)

    # TODO see if this is the correct order
    time_differences = distance_score.indexes["t"] - target_date
    sigma_doy = get_from_context(context, "sigma_doy")

    doy_score = np.exp(
        -1
        / 2
        * (np.array(time_differences.days) ** 2 / sigma_doy**2)
    )
    # TODO this division is likely redundant
    #doy_score = doy_score / np.max(doy_score)

    inspect(distance_score.dims, message="distance score dims")
    inspect(doy_score.shape, message="doy score dims")
    score = distance_score * doy_score.reshape((-1, 1, 1, 1))
    score = score / (np.sum(score, axis=0) + 1e-5)
    inspect(score.shape)

    score = xr.DataArray(
            score,
            #dims=["t", "x", "y", "bands"],
            dims=["t", "bands", "y", "x"],
            coords={
                "t": distance_score["t"],
                "x": distance_score["x"],
                "y": distance_score["y"],
                "bands": ["weight"]
            }
        )

    return XarrayDataCube(score)


def get_from_context(context, key):
    return context.get(key, CONTEXT[key])
