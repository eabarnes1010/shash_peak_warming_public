"""Functions for working with generic files.

Functions
---------
get_model_name(settings)
get_netcdf_da(filename)
save_pred_obs(pred_vector, filename)
save_tf_model(model, model_name, directory, settings)
get_cmip_filenames(settings, verbose=0)
"""

import xarray as xr
import numpy as np


__author__ = "Elizabeth A. Barnes and Noah Diffenbaugh"
__version__ = "05 February 2024"


def get_model_name(settings):
    model_name = settings["exp_name"] + "_rng_seed" + str(settings["rng_seed"])

    return model_name


def get_netcdf_da(filename):
    da = xr.open_dataarray(filename)
    return da


def convert_to_cftime(da, orig_time):
    da = da.rename({orig_time: "time"})
    dates = xr.cftime_range(start="1850", periods=da.shape[0], freq="YS", calendar="noleap")
    da = da.assign_coords({"time": ("time", dates, {"units": "years since 1850-01-01"})})
    return da


def get_gcm_name(f, ssp):

    istart = f.find(ssp)
    iend_a = f.find("_ann_mean")
    iend_b = f.find("_ncecat")

    if iend_a == -1:
        iend_a = 100_000_000
    if iend_b == -1:
        iend_b = 100_000_000

    return f[istart:np.min([iend_a, iend_b])]


def get_observations_filename(source, verbose=True):
    if source == "BEST":
        # nc_filename_obs = "_Land_and_Ocean_LatLong1_185001_202312_ann_mean_2pt5degree.nc"
        nc_filename_obs = "_Land_and_Ocean_LatLong1_185001_202312_anomalies_ann_mean_2pt5degree.nc"
    elif source == "GISTEMP":
        nc_filename_obs = "_gistemp1200_GHCNv4_ERSSTv5_188001_202312_ann_mean_2pt5degree.nc"
    elif source == "ERA5":
        raise NotImplementedError
        nc_filename_obs = "_ERA5_t2m_mon_194001-202311_194001_202212_ann_mean_2pt5degree.nc"
    else:
        raise NotImplementedError()

    if verbose:
        print(nc_filename_obs)

    return nc_filename_obs


def get_emissions_filename():
    """Returns emissions filename and years for the file."""
    return "global_carbon_budget_1750_2021.csv", np.arange(1750, 2022)


def get_cmip_filenames(ssp, sub, verbose=False):
    main_dict = filename_lookup_dict()

    filenames = []
    filenames.extend(main_dict[ssp][sub])

    if verbose:
        print(filenames)

    return filenames


def filename_lookup_dict():
    main_dict = {}

    # SSP1-1.9
    d_child = {}
    d_child["multi_member"] = (
        "tas_Amon_historical_ssp119_CanESM5_r1-5_ncecat_ann_mean_2pt5degree.nc",
        "tas_Amon_historical_ssp119_CNRM-ESM2-1_r1-5_ncecat_ann_mean_2pt5degree.nc",
        "tas_Amon_historical_ssp119_GISS-E2-1-G_r1-5_ncecat_ann_mean_2pt5degree.nc",
        "tas_Amon_historical_ssp119_IPSL-CM6A-LR_r1-5_ncecat_ann_mean_2pt5degree.nc",
        "tas_Amon_historical_ssp119_MIROC-ES2L_r1-5_ncecat_ann_mean_2pt5degree.nc",
        "tas_Amon_historical_ssp119_MPI-ESM1-2-LR_r1-5_ncecat_ann_mean_2pt5degree.nc",
        "tas_Amon_historical_ssp119_MRI-ESM2-0_r1-5_ncecat_ann_mean_2pt5degree.nc",
        "tas_Amon_historical_ssp119_UKESM1-0-LL_r1-5_ncecat_ann_mean_2pt5degree.nc",
    )
    d_child["single_member"] = (
        "tas_Amon_historical_ssp119_CanESM5_r1i1p1f1_ann_mean_2pt5degree.nc",
        "tas_Amon_historical_ssp119_CNRM-ESM2-1_r1i1p1f2_ann_mean_2pt5degree.nc",
        "tas_Amon_historical_ssp119_FGOALS-g3_r1i1p1f1_ann_mean_2pt5degree.nc",
        "tas_Amon_historical_ssp119_GISS-E2-1-G_r1i1p1f2_ann_mean_2pt5degree.nc",
        "tas_Amon_historical_ssp119_IPSL-CM6A-LR_r1i1p1f1_ann_mean_2pt5degree.nc",
        "tas_Amon_historical_ssp119_MIROC-ES2L_r1i1p1f2_ann_mean_2pt5degree.nc",
        "tas_Amon_historical_ssp119_MIROC6_r1i1p1f1_ann_mean_2pt5degree.nc",
        "tas_Amon_historical_ssp119_MPI-ESM1-2-LR_r1i1p1f1_ann_mean_2pt5degree.nc",
        "tas_Amon_historical_ssp119_MRI-ESM2-0_r1i1p1f1_ann_mean_2pt5degree.nc",
        "tas_Amon_historical_ssp119_UKESM1-0-LL_r1i1p1f2_ann_mean_2pt5degree.nc",
    )
    main_dict["ssp119"] = d_child

    # SSP 1-2.6
    d_child = {}
    d_child["multi_member"] = (
        "tas_Amon_historical_ssp126_MIROC6_r1-10_ncecat_ann_mean_2pt5degree.nc",
        "tas_Amon_historical_ssp126_MIROC-ES2L_r1-10_ncecat_ann_mean_2pt5degree.nc",
        "tas_Amon_historical_ssp126_MRI-ESM2-0_r1-5_ncecat_ann_mean_2pt5degree.nc",
        "tas_Amon_historical_ssp126_CanESM5_r1-10_ncecat_ann_mean_2pt5degree.nc",
        "tas_Amon_historical_ssp126_ACCESS-ESM1-5_r1-10_ncecat_ann_mean_2pt5degree.nc",
        "tas_Amon_historical_ssp126_UKESM1-0-LL_r1-10_ncecat_ann_mean_2pt5degree.nc",
        "tas_Amon_historical_ssp126_CNRM-CM6-1_r1-5_ncecat_ann_mean_2pt5degree.nc",
        "tas_Amon_historical_ssp126_CNRM-ESM2-1_r1-5_ncecat_ann_mean_2pt5degree.nc",
        "tas_Amon_historical_ssp126_GISS-E2-1-G_r1-5_ncecat_ann_mean_2pt5degree.nc",
        "tas_Amon_historical_ssp126_IPSL-CM6A-LR_r1-5_ncecat_ann_mean_2pt5degree.nc",
    )
    d_child["single_member"] = (
        "tas_Amon_historical_ssp126_ACCESS-CM2_r1i1p1f1_ann_mean_2pt5degree.nc",
        "tas_Amon_historical_ssp126_ACCESS-ESM1-5_r1i1p1f1_ann_mean_2pt5degree.nc",
        "tas_Amon_historical_ssp126_BCC-CSM2-MR_r1i1p1f1_ann_mean_2pt5degree.nc",
        "tas_Amon_historical_ssp126_CanESM5_r1i1p1f1_ann_mean_2pt5degree.nc",
        "tas_Amon_historical_ssp126_CAS-ESM2-0_r1i1p1f1_ann_mean_2pt5degree.nc",
        "tas_Amon_historical_ssp126_CESM2_r4i1p1f1_ann_mean_2pt5degree.nc",
        "tas_Amon_historical_ssp126_CMCC-CM2-SR5_r1i1p1f1_ann_mean_2pt5degree.nc",
        "tas_Amon_historical_ssp126_CMCC-ESM2_r1i1p1f1_ann_mean_2pt5degree.nc",
        "tas_Amon_historical_ssp126_CNRM-CM6-1_r1i1p1f2_ann_mean_2pt5degree.nc",
        "tas_Amon_historical_ssp126_CNRM-CM6-1-HR_r1i1p1f2_ann_mean_2pt5degree.nc",
        "tas_Amon_historical_ssp126_CNRM-ESM2-1_r1i1p1f2_ann_mean_2pt5degree.nc",
        "tas_Amon_historical_ssp126_FGOALS-f3-L_r1i1p1f1_ann_mean_2pt5degree.nc",
        "tas_Amon_historical_ssp126_FGOALS-g3_r1i1p1f1_ann_mean_2pt5degree.nc",
        "tas_Amon_historical_ssp126_GFDL-ESM4_r1i1p1f1_ann_mean_2pt5degree.nc",
        "tas_Amon_historical_ssp126_GISS-E2-1-G_r1i1p1f2_ann_mean_2pt5degree.nc",
        "tas_Amon_historical_ssp126_HadGEM3-GC31-LL_r1i1p1f3_ann_mean_2pt5degree.nc",
        "tas_Amon_historical_ssp126_HadGEM3-GC31-MM_r1i1p1f3_ann_mean_2pt5degree.nc",
        "tas_Amon_historical_ssp126_INM-CM4-8_r1i1p1f1_ann_mean_2pt5degree.nc",
        "tas_Amon_historical_ssp126_INM-CM5-0_r1i1p1f1_ann_mean_2pt5degree.nc",
        "tas_Amon_historical_ssp126_IPSL-CM6A-LR_r1i1p1f1_ann_mean_2pt5degree.nc",
        "tas_Amon_historical_ssp126_KACE-1-0-G_r1i1p1f1_ann_mean_2pt5degree.nc",
        "tas_Amon_historical_ssp126_KIOST-ESM_r1i1p1f1_ann_mean_2pt5degree.nc",
        "tas_Amon_historical_ssp126_MIROC-ES2L_r1i1p1f2_ann_mean_2pt5degree.nc",
        "tas_Amon_historical_ssp126_MIROC6_r1i1p1f1_ann_mean_2pt5degree.nc",
        "tas_Amon_historical_ssp126_MRI-ESM2-0_r1i1p1f1_ann_mean_2pt5degree.nc",
        "tas_Amon_historical_ssp126_NESM3_r1i1p1f1_ann_mean_2pt5degree.nc",
        "tas_Amon_historical_ssp126_NorESM2-LM_r1i1p1f1_ann_mean_2pt5degree.nc",
        "tas_Amon_historical_ssp126_NorESM2-MM_r1i1p1f1_ann_mean_2pt5degree.nc",
        "tas_Amon_historical_ssp126_TaiESM1_r1i1p1f1_ann_mean_2pt5degree.nc",
        "tas_Amon_historical_ssp126_UKESM1-0-LL_r1i1p1f2_ann_mean_2pt5degree.nc",
    )
    main_dict["ssp126"] = d_child

    # SSP 2-4.5
    d_child = {}
    d_child["multi_member"] = (
        "tas_Amon_historical_ssp245_ACCESS-ESM1-5_r1-10_ncecat_ann_mean_2pt5degree.nc",
        "tas_Amon_historical_ssp245_CanESM5_r1-10_ncecat_ann_mean_2pt5degree.nc",
        "tas_Amon_historical_ssp245_CNRM-ESM2-1_r1-10_ncecat_ann_mean_2pt5degree.nc",
        "tas_Amon_historical_ssp245_GISS-E2-1-G_r1-10_ncecat_ann_mean_2pt5degree.nc",
        "tas_Amon_historical_ssp245_IPSL-CM6A-LR_r1-10_ncecat_ann_mean_2pt5degree.nc",
        "tas_Amon_historical_ssp245_MIROC-ES2L_r1-10_ncecat_ann_mean_2pt5degree.nc",
    )
    d_child["single_member"] = (
        "tas_Amon_historical_ssp245_ACCESS-CM2_r1i1p1f1_ann_mean_2pt5degree.nc",
        "tas_Amon_historical_ssp245_ACCESS-ESM1-5_r1i1p1f1_ann_mean_2pt5degree.nc",
        "tas_Amon_historical_ssp245_AWI-CM-1-1-MR_r1i1p1f1_ann_mean_2pt5degree.nc",
        "tas_Amon_historical_ssp245_BCC-CSM2-MR_r1i1p1f1_ann_mean_2pt5degree.nc",
        "tas_Amon_historical_ssp245_CanESM5_r1i1p1f1_ann_mean_2pt5degree.nc",
        "tas_Amon_historical_ssp245_CAS-ESM2-0_r1i1p1f1_ann_mean_2pt5degree.nc",
        "tas_Amon_historical_ssp245_CESM2_r4i1p1f1_ann_mean_2pt5degree.nc",
        "tas_Amon_historical_ssp245_CESM2-WACCM_r1i1p1f1_ann_mean_2pt5degree.nc",
        "tas_Amon_historical_ssp245_CMCC-CM2-SR5_r1i1p1f1_ann_mean_2pt5degree.nc",
        "tas_Amon_historical_ssp245_CMCC-ESM2_r1i1p1f1_ann_mean_2pt5degree.nc",
        "tas_Amon_historical_ssp245_CNRM-CM6-1_r1i1p1f2_ann_mean_2pt5degree.nc",
        "tas_Amon_historical_ssp245_CNRM-CM6-1-HR_r1i1p1f2_ann_mean_2pt5degree.nc",
        "tas_Amon_historical_ssp245_CNRM-ESM2-1_r1i1p1f2_ann_mean_2pt5degree.nc",
        "tas_Amon_historical_ssp245_FGOALS-f3-L_r1i1p1f1_ann_mean_2pt5degree.nc",
        "tas_Amon_historical_ssp245_FGOALS-g3_r1i1p1f1_ann_mean_2pt5degree.nc",
        "tas_Amon_historical_ssp245_GFDL-CM4_r1i1p1f1_ann_mean_2pt5degree.nc",
        "tas_Amon_historical_ssp245_GFDL-ESM4_r1i1p1f1_ann_mean_2pt5degree.nc",
        "tas_Amon_historical_ssp245_GISS-E2-1-G_r1i1p1f2_ann_mean_2pt5degree.nc",
        "tas_Amon_historical_ssp245_IPSL-CM6A-LR_r1i1p1f1_ann_mean_2pt5degree.nc",
        "tas_Amon_historical_ssp245_MIROC-ES2L_r1i1p1f2_ann_mean_2pt5degree.nc",
        "tas_Amon_historical_ssp245_MIROC6_r1i1p1f1_ann_mean_2pt5degree.nc",
        "tas_Amon_historical_ssp245_MRI-ESM2-0_r1i1p1f1_ann_mean_2pt5degree.nc",
        "tas_Amon_historical_ssp245_NorESM2-LM_r1i1p1f1_ann_mean_2pt5degree.nc",
    )
    main_dict["ssp245"] = d_child

    # SSP 3-7.0
    d_child = {}
    d_child["multi_member"] = (
        "tas_Amon_historical_ssp370_ACCESS-ESM1-5_r1-10_ncecat_ann_mean_2pt5degree.nc",
        "tas_Amon_historical_ssp370_CanESM5_r1-10_ncecat_ann_mean_2pt5degree.nc",
        "tas_Amon_historical_ssp370_CESM2-LE2-smbb_r1-10_ncecat_ann_mean_2pt5degree.nc",
        "tas_Amon_historical_ssp370_GISS-E2-1-G_r1-10_ncecat_ann_mean_2pt5degree.nc",
        "tas_Amon_historical_ssp370_IPSL-CM6A-LR_r1-10_ncecat_ann_mean_2pt5degree.nc",
        "tas_Amon_historical_ssp370_MIROC-ES2L_r1-10_ncecat_ann_mean_2pt5degree.nc",
        "tas_Amon_historical_ssp370_UKESM1-0-LL_r1-10_ncecat_ann_mean_2pt5degree.nc",
    )
    d_child["single_member"] = ()
    main_dict["ssp370"] = d_child

    return main_dict
