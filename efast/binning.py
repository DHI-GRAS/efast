from collections import namedtuple
from pathlib import Path
from typing import Iterable,  Self
import warnings
from numpy.typing import NDArray
import xarray as xr
import re
import numpy as np
from scipy import stats
import shapely
import cv2
import rasterio
from rasterio.transform import Affine

from efast.constants import S3L2SYNClassificationAerosolFlags, S3L2SYNCloudFlags


FILL_VALUE = -10000 # TODO move

CLOUD_FLAGS_COMBINED = (
    S3L2SYNCloudFlags.CLOUD
    | S3L2SYNCloudFlags.CLOUD_AMBIGUOUS
    | S3L2SYNCloudFlags.CLOUD_MARGIN
)

def binning_s3_py(
    download_dir,
    binning_dir,
    footprint,
    s3_bands=["SDR_Oa04", "SDR_Oa06", "SDR_Oa08", "SDR_Oa17"],
    **kwargs
):
    """
    TODO: "binning" might be a misnomer, as the function does more than just binning
    """
    band_names = [*s3_bands, "CLOUD_flags", "SYN_flags"]
    sen3_paths = list(download_dir.glob("*.SEN3"))
    bbox = BBox.from_wkt(footprint)
    grid = create_geogrid(bbox, num_rows=66792)
    pixel_size = grid.lat[1] - grid.lat[0]
    transform = Affine.translation(grid.lon[0], grid.lat[-1]) * Affine.scale(pixel_size, -pixel_size)

    for i, sen3_path in enumerate(sen3_paths):
        output_path = binning_dir / (sen3_path.stem + ".tif")

        prod = SynProduct(sen3_path, band_names=band_names)
        ds = prod.read_bands()

        cloud_no_land_mask = ((ds["CLOUD_flags"] & CLOUD_FLAGS_COMBINED.value) != 0) & (ds["SYN_flags"] & S3L2SYNClassificationAerosolFlags.SYN_land.value > 0)

        for reflectance_band in s3_bands:
             band_values = ds[reflectance_band].data
             band_values[cloud_no_land_mask] = -10000
             ds[reflectance_band] = (["lat", "lon"], band_values)


        binned = bin_to_grid_numpy(ds, s3_bands, grid, super_sampling=2)

        binned[binned < 0] = np.nan


        with rasterio.open(output_path, "w", driver="GTiff", height=binned.shape[1], width=binned.shape[2], count=len(s3_bands), dtype=binned.dtype, crs="+proj=latlong", transform=transform, nodata=np.nan) as ds:
            ds.write(binned[:, ::-1, :])


def get_reflectance_filename(index: int):
    if not (0 < index <= 21):
        raise ValueError(
            f"The index must be an integer between 1 and 21 (both inclusive)"
        )

    return f"SDR_Oa{index:02}"


GEOLOCATION_FILE_NAME = "geolocation.nc"
FLAG_FILE_NAME = "flags.nc"

SCALE_FACTOR = 1e-4  # TODO read from netcdf


# TODO better assume that band_names are already actual band names. Calculate which netcdf file you need independently.
class SynProduct:
    FLAG_VARIABLE_NAMES = [
        "CLOUD_flags",
        "OLC_flags",
        "SLN_flags",
        "SLO_flags",
        "SYN_flags",
    ]

    def __init__(self, path: Path | str, band_names: list[str]):
        self.path = Path(path)
        self.band_names = band_names

    def read_bands(self) -> xr.Dataset:
        # sort bands into flag bands, reflectance bands, others
        flag_band_names = list(filter(lambda b: b.endswith("_flags"), self.band_names))
        reflectance_band_names = list(
            filter(lambda b: b.startswith("SDR_Oa"), self.band_names)
        )
        remaining_band_names = [
            b
            for b in self.band_names
            if b not in set([*flag_band_names, *reflectance_band_names])
        ]

        if len(remaining_band_names) != 0:
            raise ValueError(
                f"Band names '{remaining_band_names}' are neither "
                "flags nor reflectance bands. Cannot open."
            )

        # geolocation
        geolocation_filename = self.path / GEOLOCATION_FILE_NAME
        geolocation_ds = xr.open_dataset(geolocation_filename)
        lat = geolocation_ds["lat"].data
        lon = geolocation_ds["lon"].data

        # reflectance bands
        reflectance_bands = self.open_reflectance_bands(reflectance_band_names)

        # flag bands
        flag_bands = self.open_flag_bands(flag_band_names)

        dims = ["lat", "lon"]
        bands = {**reflectance_bands, **flag_bands}
        bands = {name: (dims, band.data) for (name, band) in bands.items()}
        bands["lat"] = (["x", "y"], lat)
        bands["lon"] = (["x", "y"], lon)
        # join
        ds = xr.Dataset(
            bands,
        )
        ds.set_coords(("lat", "lon"))

        return ds

    def open_reflectance_bands(self, band_names: list[str]) -> dict[str, xr.DataArray]:
        bands: list[xr.DataArray] = []
        for band_name in band_names:
            file_name = determine_file_name_from_reflectance_variable_name(band_name)
            # by default `mask_and_scale=None` behaves as if `mask_and_scale=True`
            band_ds = xr.open_dataset(self.path / file_name, mask_and_scale=False)
            bands.append(band_ds[band_name])

        return {name: band for (name, band) in zip(band_names, bands)}

    def open_flag_bands(self, band_names: list[str]) -> dict[str, xr.DataArray]:
        flag_ds = xr.open_dataset(self.path / FLAG_FILE_NAME)

        band_name_exists = [bn in flag_ds.variables.keys() for bn in band_names]
        nonexistent_bands = [
            band for (band, exists) in zip(band_names, band_name_exists) if not exists
        ]

        if len(nonexistent_bands) > 0:
            raise ValueError(
                f"Could not find bands '{nonexistent_bands}' in file {self.path / FLAG_FILE_NAME}"
            )

        return {bn: flag_ds[bn] for bn in band_names}


def determine_file_name_from_reflectance_variable_name(varname: str):
    index_group = 1
    pattern = f"SDR_Oa(..)"
    m = re.match(pattern, varname)
    if m is None:
        raise ValueError(
            f"variable name '{varname}' does not match pattern '{pattern}'. Not a reflectance_band."
        )

    return f"Syn_Oa{int(m.group(index_group)):02}_reflectance.nc"  # pyright: ignore [reportOptionalMemberAccess]


class BBox:
    def __init__(
        self, *, lat_min: float, lat_max: float, lon_min: float, lon_max: float
    ) -> None:
        self.lat_min = lat_min
        self.lat_max = lat_max
        self.lon_min = lon_min
        self.lon_max = lon_max

    @classmethod
    def from_wkt(cls, wkt: str) -> Self:
        geom = shapely.from_wkt(wkt)
        envelope = shapely.envelope(geom)
        lon_min, lat_min, lon_max, lat_max = envelope.bounds
        return cls(
            lat_min=lat_min,
            lat_max=lat_max,
            lon_min=lon_min,
            lon_max=lon_max,
        )


Grid = namedtuple("Grid", ["lat", "lon"])

def bin_to_grid(ds: xr.Dataset, bands: Iterable[str], grid: Grid,*, super_sampling: int=1, interpolation_order: int=1) -> NDArray:
    lat = ds["lat"]
    lon = ds["lon"]

    #lat = ndimage.zoom(lat, super_sampling, order=interpolation_order).ravel()
    # lon = ndimage.zoom(lon, super_sampling, order=interpolation_order).ravel()
    lat = super_sample_opencv(lat.to_numpy(), super_sampling, interpolation=cv2.INTER_LINEAR)
    lon = super_sample_opencv(lon.to_numpy(), super_sampling, interpolation=cv2.INTER_LINEAR)

    binned = []

    for band in bands:
        data = ds[band].data
        if super_sampling != 1:
            kernel = np.ones((super_sampling, super_sampling))

            data = super_sample(data, super_sampling)
        res = stats.binned_statistic_2d(
            lat,
            lon,
            values=data.ravel(),
            statistic="mean",
            bins=(grid.lat, grid.lon),  # definition of target grid
            #range=bbox,
        )
        binned.append(res.statistic)

    binned = np.array(binned)
    return binned

def bin_to_grid_numpy(ds: xr.Dataset, bands: Iterable[str], grid: Grid,*, super_sampling: int=1, interpolation_order: int=1) -> NDArray:
    lat2d = ds["lat"].data
    lon2d = ds["lon"].data

    lat = super_sample_opencv(lat2d, super_sampling, interpolation=cv2.INTER_LINEAR).ravel()
    lon = super_sample_opencv(lon2d, super_sampling, interpolation=cv2.INTER_LINEAR).ravel()

    width = grid.lon.shape[0] - 1
    height = grid.lat.shape[0] - 1

    pixel_size = (grid.lon[-1] - grid.lon[0]) / width

    # Reuse for outputs
    bin_idx_buf = np.zeros_like(lat)

    bin_idx_row = np.divide((lat - grid.lat[0]),  pixel_size, out=bin_idx_buf).astype(int)
    bin_idx_col = np.divide((lon - grid.lon[0]), pixel_size, out=bin_idx_buf).astype(int)

    bin_idx = bin_idx_row * width + bin_idx_col
    
    bin_idx[(bin_idx_row < 0) | (bin_idx_row > height) | (bin_idx_col < 0) | (bin_idx_col > width)] = -1

    counts, _ = np.histogram(bin_idx, width * height, range=(0, width*height))

    binned = []
    means = None
    sampled_data = None if super_sampling == 1 else np.zeros((lat2d.shape[0] * super_sampling, lat2d.shape[1] * super_sampling), dtype=np.int16)

    for band in bands:
        data = ds[band].data
        #data[data == FILL_VALUE] = 0
        if super_sampling != 1:
            super_sample(data, super_sampling, out=sampled_data)
        else:
            sampled_data = data

        # Promote datatype to avoid overflows
        hist, _ = np.histogram(bin_idx, range(width * height + 1), weights=sampled_data.astype(np.int32).reshape(-1), range=(0, width*height))
        means = np.divide(hist, counts, out=means)
        scaled_means = means.reshape(height, width) * SCALE_FACTOR
        binned.append(scaled_means)

    binned = np.array(binned)
    return binned


def create_geogrid(bbox: BBox, num_rows: int = 66792):
    # -90 and 90 are included, 0 also included. Lat has one more entry than num_rows, rows are defined by the spaces between lat entries
    lat = np.linspace(0, 180, num=num_rows + 1, endpoint=True) - 90

    # 180 and 0 are included, -180 is not.
    # lon has 2 * num_rows entries
    # the last bin is between lon[-1] and lon[0] (antimeridian)
    lon = np.linspace(0, 360, num=num_rows * 2 + 1, endpoint=True) - 180
    lon = lon[1:]

    # one lat bound before first bound that is larger than the bounding box min
    # TODO I can calculate lat_idx_min (and the others) directly 
    lat_idx_min = np.argmax(lat >= bbox.lat_min) - 1
    # first lat bound that is larger than the bounding box max
    lat_idx_max = np.argmax(lat > bbox.lat_max)
    # one lon bound before first bound that is larger than the bounding box min
    lon_idx_min = np.argmax(lon >= bbox.lon_min) - 1
    # first lon bound that is larger than the bounding box max
    lon_idx_max = np.argmax(lon > bbox.lon_max)

    grid = Grid(lat=lat[lat_idx_min:lat_idx_max+1], lon=lon[lon_idx_min:lon_idx_max+1])
    return grid


def super_sample(arr, factor, *, out=None):
    #return super_sample_kron(arr, factor, out=out)
    #return super_sample_repeat(arr, factor, out=out)
    return super_sample_opencv(arr, factor, out=out)

def super_sample_kron(arr, factor, *, out=None):
    if out is not None:
        warnings.warn("Parameter 'out' not supported for kron super sampling")
    kernel = np.ones((factor, factor))
    return np.kron(arr, kernel)

def super_sample_repeat(arr, factor, *, out=None):
    if out is not None:
        warnings.warn("Parameter 'out' not supported for repeat super sampling")
    return arr.repeat(factor, axis=1).repeat(factor, axis=0)


def super_sample_opencv(arr, factor,*, out=None, interpolation=cv2.INTER_NEAREST):
    if out is None:
        out = np.zeros((arr.shape[0] * factor, arr.shape[1] * factor), dtype=arr.dtype)

    res = cv2.resize(arr, dst=out, dsize=out.shape[::-1], fx=2, fy=2, interpolation=interpolation)
    return out
