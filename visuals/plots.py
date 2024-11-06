"""Metrics for generic plotting.

Functions
---------
savefig(filename, fig_format=(".png", ".pdf"), dpi=300)
adjust_spines(ax, spines)
format_spines(ax)
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import cartopy as ct
import numpy.ma as ma
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cmaps as cmaps_ncl
import regionmask
import matplotlib.colors as mcolors
import gc
from scipy.optimize import curve_fit
from matplotlib import colors

from shash.shash_torch import Shash
import model.metric as module_metric


mpl.rcParams["figure.facecolor"] = "white"
mpl.rcParams["figure.dpi"] = 150

FS = 10
plt.rc("text", usetex=False)
plt.rc("font", **{"family": "sans-serif", "sans-serif": ["Helvetica"]})
plt.rc("savefig", facecolor="white")
plt.rc("axes", facecolor="white")
plt.rc("axes", labelcolor="dimgrey")
plt.rc("axes", labelcolor="dimgrey")
plt.rc("xtick", color="dimgrey")
plt.rc("ytick", color="dimgrey")


def savefig(filename, fig_format=(".png", ".pdf"), dpi=300):
    for fig_format in fig_format:
        plt.savefig(filename + fig_format, bbox_inches="tight", dpi=dpi)


def adjust_spines(ax, spines):
    for loc, spine in ax.spines.items():
        if loc in spines:
            spine.set_position(("outward", 5))
        else:
            spine.set_color("none")
    if "left" in spines:
        ax.yaxis.set_ticks_position("left")
    else:
        ax.yaxis.set_ticks([])
    if "bottom" in spines:
        ax.xaxis.set_ticks_position("bottom")
    else:
        ax.xaxis.set_ticks([])


def format_spines(ax):
    adjust_spines(ax, ["left", "bottom"])
    ax.spines["top"].set_color("none")
    ax.spines["right"].set_color("none")
    ax.spines["left"].set_color("dimgrey")
    ax.spines["bottom"].set_color("dimgrey")
    ax.spines["left"].set_linewidth(2)
    ax.spines["bottom"].set_linewidth(2)
    ax.tick_params("both", length=4, width=2, which="major", color="dimgrey")


def get_discrete_colornorm(cb_bounds, cmap):
    cb_n = int((cb_bounds[1] - cb_bounds[0]) / cb_bounds[-1])
    # cbar_n = (cb_bounds[1] - cb_bounds[-1]) - (cb_bounds[0] - cb_bounds[-1])
    clr_norm = colors.BoundaryNorm(
        np.linspace(
            cb_bounds[0] - cb_bounds[-1] / 2, cb_bounds[1] + cb_bounds[-1] / 2, cb_n + 2
        ),
        cmap.N,
    )

    return clr_norm


def plot_one_to_one_diagnostic(
    output_val,
    output_test,
    target_val,
    target_test,
    yrs_test,
):
    pr_val = Shash(output_val).median().numpy()
    pr_test = Shash(output_test).median().numpy()

    lowerbound_test = Shash(output_test).quantile(0.25).numpy()
    upperbound_test = Shash(output_test).quantile(0.75).numpy()

    mae_test = module_metric.custom_mae(output_test, target_test)

    # --------------------------------
    plt.subplots(1, 2, figsize=(15, 6))

    plt.subplot(1, 2, 1)
    plt.plot(
        target_val,
        pr_val,
        ".",
        label="validation",
        color="gray",
        alpha=0.75,
    )

    plt.errorbar(
        target_test,
        pr_test,
        yerr=np.concatenate(
            (
                pr_test - lowerbound_test[np.newaxis, :],
                upperbound_test[np.newaxis, :] - pr_test,
            ),
            axis=0,
        ),
        linewidth=0.5,
        color="tab:purple",
        alpha=0.5,
        marker=".",
        linestyle="none",
        label="testing",
    )

    plt.axvline(x=0, color="gray", linewidth=1)
    plt.axhline(y=0, color="gray", linewidth=1)
    plt.title("Testing MAE = " + str(np.round(mae_test, 2)) + " C")
    plt.xlabel("true number of degrees left to warm")
    plt.ylabel("predicted degrees left to warm")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.errorbar(
        yrs_test,
        pr_test,
        yerr=np.concatenate(
            (
                pr_test - lowerbound_test[np.newaxis, :],
                upperbound_test[np.newaxis, :] - pr_test,
            ),
            axis=0,
        ),
        marker=".",
        linestyle="none",
        linewidth=0.5,
        color="tab:purple",
        alpha=0.5,
        label="testing",
    )

    plt.legend()
    plt.title("Degrees left to warm")
    plt.xlabel("year of map")
    plt.ylabel("predicted degrees left to warm")
    plt.axhline(y=0, color="gray", linewidth=1)


def plot_metrics_panels(trainer, config):
    plt.figure(figsize=(20, 4))
    for i, m in enumerate(("loss", *config["metrics"])):
        plt.subplot(1, 4, i + 1)
        plt.plot(trainer.log.history["epoch"], trainer.log.history[m], label=m)
        plt.plot(
            trainer.log.history["epoch"],
            trainer.log.history["val_" + m],
            label="val_" + m,
        )
        plt.axvline(
            x=trainer.early_stopper.best_epoch,
            linestyle="--",
            color="k",
            linewidth=0.75,
        )
        plt.title(m)
        plt.legend()
        plt.xlabel("epoch")


def plot_map(x, clim=None, title=None, text=None, cmap="RdGy"):
    plt.pcolor(
        x,
        cmap=cmap,
    )
    plt.clim(clim)
    plt.colorbar()
    plt.title(title, fontsize=15, loc="right")
    plt.yticks([])
    plt.xticks([])

    plt.text(
        0.01,
        1.0,
        text,
        fontfamily="monospace",
        fontsize="small",
        va="bottom",
        transform=plt.gca().transAxes,
    )


def drawOnGlobe(
    ax,
    map_proj,
    data,
    lats,
    lons,
    cmap="coolwarm",
    vmin=None,
    vmax=None,
    inc=None,
    cbarBool=True,
    contourMap=[],
    contourVals=[],
    fastBool=False,
    extent="both",
):
    data_crs = ct.crs.PlateCarree()
    data_cyc, lons_cyc = add_cyclic_point(
        data, coord=lons
    )  # fixes white line by adding point#data,lons#ct.util.add_cyclic_point(data, coord=lons) #fixes white line by adding point
    data_cyc = data
    lons_cyc = lons

    #     ax.set_global()
    #     ax.coastlines(linewidth = 1.2, color='black')
    #     ax.add_feature(cartopy.feature.LAND, zorder=0, scale = '50m', edgecolor='black', facecolor='black')
    land_feature = cfeature.NaturalEarthFeature(
        category="physical",
        name="land",
        scale="50m",
        facecolor="None",
        edgecolor="k",
        linewidth=0.5,
    )
    ax.add_feature(land_feature)
    #     ax.GeoAxes.patch.set_facecolor('black')

    if fastBool:
        image = ax.pcolormesh(lons_cyc, lats, data_cyc, transform=data_crs, cmap=cmap)
    #         image = ax.contourf(lons_cyc, lats, data_cyc, np.linspace(0,vmax,20),transform=data_crs, cmap=cmap)
    else:
        image = ax.pcolor(
            lons_cyc, lats, data_cyc, transform=data_crs, cmap=cmap, shading="auto"
        )

    if np.size(contourMap) != 0:
        contourMap_cyc, __ = add_cyclic_point(
            contourMap, coord=lons
        )  # fixes white line by adding point
        ax.contour(
            lons_cyc,
            lats,
            contourMap_cyc,
            contourVals,
            transform=data_crs,
            colors="fuchsia",
        )

    if cbarBool:
        cb = plt.colorbar(
            image, shrink=0.45, orientation="horizontal", pad=0.02, extend=extent
        )
        cb.ax.tick_params(labelsize=6)
    else:
        cb = None

    image.set_clim(vmin, vmax)

    return cb, image


def add_cyclic_point(data, coord=None, axis=-1):
    # had issues with cartopy finding utils so copied for myself

    if coord is not None:
        if coord.ndim != 1:
            raise ValueError("The coordinate must be 1-dimensional.")
        if len(coord) != data.shape[axis]:
            raise ValueError(
                "The length of the coordinate does not match "
                "the size of the corresponding dimension of "
                "the data array: len(coord) = {}, "
                "data.shape[{}] = {}.".format(len(coord), axis, data.shape[axis])
            )
        delta_coord = np.diff(coord)
        if not np.allclose(delta_coord, delta_coord[0]):
            raise ValueError("The coordinate must be equally spaced.")
        new_coord = ma.concatenate((coord, coord[-1:] + delta_coord[0]))
    slicer = [slice(None)] * data.ndim
    try:
        slicer[axis] = slice(0, 1)
    except IndexError:
        raise ValueError(
            "The specified axis does not correspond to an array dimension."
        )
    new_data = ma.concatenate((data, data[tuple(slicer)]), axis=axis)
    if coord is None:
        return_value = new_data
    else:
        return_value = new_data, new_coord
    return return_value


def plot_pits(output, target):
    plt.figure(figsize=(8, 5))
    ax = plt.subplot(1, 1, 1)

    # compute PIT
    bins, hist_shash, D_shash, EDp_shash = module_metric.pit_d(output, target)

    clr_shash = "teal"
    bins_inc = bins[1] - bins[0]

    bin_add = bins_inc / 2
    bin_width = bins_inc * 0.98
    ax.bar(
        hist_shash[1][:-1] + bin_add,
        hist_shash[0],
        width=bin_width,
        color=clr_shash,
        label="SHASH",
    )

    # make the figure pretty
    ax.axhline(
        y=0.1,
        linestyle="--",
        color="k",
        linewidth=2.0,
    )
    # ax = plt.gca()
    yticks = np.around(np.arange(0, 0.55, 0.05), 2)
    plt.yticks(yticks, yticks)
    ax.set_ylim(0, 0.25)
    ax.set_xticks(bins, np.around(bins, 1))

    plt.text(
        0.0,
        np.max(ax.get_ylim()) * 0.99,
        "D statistic: "
        + str(np.round(D_shash, 4))
        + " ("
        + str(np.round(EDp_shash, 3))
        + ")",
        color=clr_shash,
        verticalalignment="top",
        fontsize=12,
    )

    ax.set_xlabel("probability integral transform")
    ax.set_ylabel("probability")


def plot_xai_heatmaps(
    xplot,
    xplot_transfer,
    xplot_cmip,
    lat,
    lon,
    ipcc_region,
    subplots=3,
    scaling=1,
    diff_scaling=1.0,
    title=None,
    colorbar=True,
):
    c = cmaps_ncl.BlueDarkRed18_r.colors
    c = np.insert(c, 9, [1, 1, 1], axis=0)
    cmap = mpl.colors.ListedColormap(c)

    transform = ct.crs.PlateCarree()
    projection = ct.crs.EqualEarth(central_longitude=0.0)

    xplot = xplot * scaling
    xplot_transfer = xplot_transfer * scaling
    if subplots == 3:
        xplot_cmip = xplot_cmip * scaling

    fig = plt.figure(figsize=(1.5 * 5.25 * 2, 1.5 * 3.25 * 1), dpi=200)

    if subplots == 3:
        a1 = fig.add_subplot(1, 3, 1, projection=projection)
        c1 = a1.pcolormesh(
            lon,
            lat,
            xplot_cmip,
            cmap=cmap,
            transform=transform,
        )
        a1.add_feature(
            cfeature.NaturalEarthFeature(
                "physical",
                "land",
                "110m",
                edgecolor="k",
                linewidth=0.5,
                facecolor="None",
            )
        )
        regionmask.defined_regions.ar6.land[(ipcc_region,)].plot(
            add_label=False,
            label_multipolygon="all",
            add_ocean=False,
            ocean_kws=dict(color="lightblue", alpha=0.25),
            line_kws=dict(
                linewidth=1.0,
            ),
        )
        c1.set_clim(-1, 1)
        if colorbar:
            fig.colorbar(
                c1,
                orientation="horizontal",
                shrink=0.35,
                extend="both",
                pad=0.02,
            )
        if title is not None:
            plt.title("(a) CMIP6 SHAP " + title)

    a1 = fig.add_subplot(1, 3, 2, projection=projection)
    c1 = a1.pcolormesh(
        lon,
        lat,
        xplot,
        cmap=cmap,
        transform=transform,
    )
    a1.add_feature(
        cfeature.NaturalEarthFeature(
            "physical",
            "land",
            "110m",
            edgecolor="k",
            linewidth=0.5,
            facecolor="None",
        )
    )
    regionmask.defined_regions.ar6.land[(ipcc_region,)].plot(
        add_label=False,
        label_multipolygon="all",
        add_ocean=False,
        ocean_kws=dict(color="lightblue", alpha=0.25),
        line_kws=dict(
            linewidth=1.0,
        ),
    )
    c1.set_clim(-1, 1)
    if colorbar:
        fig.colorbar(
            c1,
            orientation="horizontal",
            shrink=0.35,
            extend="both",
            pad=0.02,
        )
    if title is not None:
        plt.title("(b) Observations SHAP " + title)

    a1 = fig.add_subplot(1, 3, 3, projection=projection)
    c1 = a1.pcolormesh(
        lon,
        lat,
        xplot_transfer - xplot,
        cmap=cmap,
        transform=transform,
    )
    a1.add_feature(
        cfeature.NaturalEarthFeature(
            "physical",
            "land",
            "110m",
            edgecolor="k",
            linewidth=0.5,
            facecolor="None",
        )
    )
    regionmask.defined_regions.ar6.land[(ipcc_region,)].plot(
        add_label=False,
        label_multipolygon="all",
        add_ocean=False,
        ocean_kws=dict(color="lightblue", alpha=0.25),
        line_kws=dict(
            linewidth=1.0,
        ),
    )
    c1.set_clim(-1.0 * diff_scaling, 1.0 * diff_scaling)
    if colorbar:
        fig.colorbar(
            c1,
            orientation="horizontal",
            shrink=0.35,
            extend="both",
            pad=0.02,
        )
    if title is not None:
        plt.title("(c) Transfer minus Original SHAP " + title)


def plot_predicted_max(d, median, expname):

    color = mcolors._colors_full_map
    color = list(color.values())
    rng = np.random.default_rng(seed=66)
    rng.shuffle(color)

    counter = 0
    for i, name in enumerate(np.unique(d["gcm_name"])):
        for ssp in np.unique(d["ssp"]):
            isample = np.where((d["gcm_name"] == name) & (d["ssp"] == ssp))[0]
            if len(isample) < 1:
                continue

            counter = counter + 1
            plt.plot(
                d["year"][isample],
                (median[isample] + d["current_temp"][isample]),
                ".",
                color=color[counter],
                label=f"{name}: {d['ssp'][isample[0]]}",
                alpha=0.75,
            )
            plt.axhline(
                y=d["max_temp"][isample[0]],
                color=color[counter],
                linewidth=1.25,
                linestyle="--",
                zorder=1000,
            )

    plt.legend(fontsize=4)
    plt.ylabel("degrees C above baseline")
    plt.axhline(y=0, color="gray", linewidth=0.75)
    plt.title(expname + ": predicted max temperature")


def plot_label_definition(
    year_reached,
    max_temp,
    global_mean_ens,
    global_mean,
    baseline_mean,
    anomalies,
    iyrs,
    config,
):

    # plot the calculation to make sure things make sense
    plt.figure(dpi=200)

    color = mcolors._colors_full_map
    color = list(color.values())
    rng = np.random.default_rng(seed=31415)
    rng.shuffle(color)

    for ens in np.arange(0, global_mean_ens.shape[0]):
        plt.plot(
            global_mean_ens["time.year"],
            global_mean_ens[ens, :],
            linewidth=1.0,
            color=color[ens],
            alpha=0.5,
        )
        if config["label_ensmean"] is False:
            plt.axvline(
                x=year_reached[ens],
                color=color[ens],
                linewidth=1.0,
                linestyle="-",
                label=f"#{ens}: {max_temp[ens].round(2)}C in {year_reached[ens]}",
            )
    plt.plot(
        global_mean["time.year"],
        global_mean,
        linewidth=1,
        color="k",
        alpha=0.75,
    )
    if config["label_forced"]:
        if config["label_ensmean"]:
            label = f"forced: {max_temp[0].round(2)}C in {year_reached[0]}"
            plt.axvline(
                x=year_reached[0],
                color="k",
                linewidth=1.0,
                linestyle="-",
            )
        else:
            label = None
        plt.plot(
            global_mean["time.year"][iyrs],
            (anomalies + baseline_mean.values[:, np.newaxis]).T,
            linewidth=2,
            label=label,
            color="k",
        )
    for i in range(baseline_mean.shape[0]):
        plt.axhline(
            y=baseline_mean[i],
            color="k",
            linestyle="-",
            linewidth=1,
        )
    plt.legend(fontsize=6)
    # plt.tight_layout()
