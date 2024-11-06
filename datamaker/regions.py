"""Region definitions.

Functions
---------
compute_global_mean(da)
extract_region(data, region, dir, land_only=False, lat=None, lon=None)

"""

import xarray as xr
import numpy as np
import pandas as pd
import regionmask
import matplotlib.pyplot as plt


def compute_global_mean(da, lat=[], lon=[]):

    if not isinstance(da, xr.DataArray):
        assert da.shape[-1] == len(lon)
        assert da.shape[-2] == len(lat)
        assert len(da.shape) <= 3

        if len(da.shape) == 2:
            da = da[None, :, :]

        da = xr.DataArray(
            da,
            coords=(np.arange(0, da.shape[0]), lat, lon),
            dims=("sample", "lat", "lon"),
        )

    weights = np.cos(np.deg2rad(da.lat))
    weights.name = "weights"
    temp_weighted = da.weighted(weights)
    global_mean = temp_weighted.mean(("lon", "lat"), skipna=True)

    return global_mean


def extract_region(data, region, dir, land_only=False, lat=None, lon=None):
    if region is None:
        return data, None, None

    if region == "globe":
        return data, None, None
    else:
        ar6_land = regionmask.defined_regions.ar6.land
        mask = regionmask.defined_regions.ar6.land.mask(lon, lat)

        i = ar6_land.abbrevs.index(region)
        mask_subset = np.where(mask == i, 1.0, np.nan)

        if land_only:
            shapefile_dir = dir + "shapefiles/"
            country_mask = xr.load_dataarray(shapefile_dir + "countries_10m_2.5x2.5.nc")
            mask_subset = np.where(~np.isnan(country_mask), mask_subset, np.nan)

        return data * mask_subset
