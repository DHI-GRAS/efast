import openeo


def test_main():
    connection = openeo.connect("https://openeo.dataspace.copernicus.eu/")
    connection.authenticate_oidc()

    s2_cube = connection.load_collection(
        "SENTINEL2_L2A",
        spatial_extent={"west": 4.00, "south": 51.04, "east": 4.10, "north": 51.1},
        temporal_extent=["2022-03-01", "2022-03-12"],
        bands=["B02", "B03", "B04"],
    )
    udf = openeo.UDF("""
import xarray


def apply_datacube(cube: xarray.DataArray, context: dict) -> xarray.DataArray:

    cube.values = 0.0001 * cube.values

    return cube
""")
    rescaled = s2_cube.apply(process=udf)
    rescaled.download("apply-udf-scaling.nc")
