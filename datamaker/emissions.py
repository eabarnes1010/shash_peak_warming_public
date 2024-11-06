"""Emissions functions for the SSPs

Functions
---------
get_emissions()
"""

__author__ = "Noah Diffenbaugh and Elizabeth A. Barnes"
__date__ = "24 January 2024"


import numpy as np
from matplotlib import pyplot as plt
import visuals.plots as plots
import pandas as pd
import datamaker.filemethods as filemethods

colors = {
    "ssp119": "#39a3c6",
    "ssp126": "#202e52",
    "ssp245": "#e78928",
    "ssp370": "#E4201D",
    "obs": "k",
}


def create_emissions(ssp, years, data_dir, figure_dir, endyear=2250, plot=False):
    emiss, emiss_years = get_emissions(ssp, data_dir, figure_dir, plot=False)

    assert (endyear >= 1850) & (endyear <= 2250)

    # zero out years after endyear
    iyear = np.where(emiss_years > endyear)[0]
    emiss[iyear] = 0.0

    # sub-select the requested years
    istart = int(np.where(emiss_years == years[0])[0])
    iend = int(np.where(emiss_years == years[-1])[0] + 1)
    emiss_years = emiss_years[istart:iend]

    # compute the cumulative emissions over those years,
    # then grab the years of the dataset
    cum_emiss_left = get_cumulative_emissions_left(emiss)
    emiss = emiss[istart:iend]
    cum_emiss_left = cum_emiss_left[istart:iend]

    # determine whether to use extended emissions in the accumulation
    # if extended_emissions:
    #     cum_emiss_left = get_cumulative_emissions_left(emiss)
    #     emiss = emiss[istart:iend]
    #     cum_emiss_left = cum_emiss_left[istart:iend]
    # else:
    #     emiss = emiss[istart:iend]
    #     cum_emiss_left = get_cumulative_emissions_left(emiss)

    if plot:
        plt.figure(figsize=(7, 4))
        for ssp in ("ssp119", "ssp126", "ssp245"):
            emiss, cum_emiss_left, emiss_years = create_emissions(
                ssp,
                years,
                data_dir,
                figure_dir,
                endyear=2250,
                plot=True,
            )

            # plot cumulative emissions left
            plt.plot(
                emiss_years,
                cum_emiss_left,
                color=colors[ssp],
                linewidth=3,
                alpha=0.8,
                label=ssp,
            )

        plt.axhline(y=0, color="k", alpha=1, linewidth=.75)
        plt.ylabel("emissions left (Gt)")
        plt.xlabel("year")
        plt.legend()
        plt.xlim(1850, 2100)

        plots.savefig(figure_dir + "data_diagnostics/cum_emissions")
        plt.close()
        raise ValueError

    return emiss, cum_emiss_left, emiss_years


def get_cumulative_emissions_left(emiss):
    total_emiss = np.sum(emiss[emiss > 0])
    return total_emiss - np.cumsum(np.abs(emiss))


def get_emissions(ssp, data_dir, figure_dir, plot=False):

    # https://gmd.copernicus.org/articles/13/3571/2020/#section7&gid=1&pid=1

    # SSP EMISSIONS
    x_ssp = [2015, 2020, 2030, 2040, 2050, 2060, 2070, 2080, 2090, 2100, 2140, 2250]
    x_interp = np.arange(1850, 2251)

    ssp119 = [
        39152.726,
        39693.726,
        22847.271,
        10475.089,
        2050.362,
        -1525.978,
        -4476.970,
        -7308.783,
        -10565.023,
        -13889.788,
        -13889.788,
        0.0,
    ]
    ssp126 = [
        39152.726,
        39804.013,
        34734.424,
        26509.183,
        17963.539,
        10527.979,
        4476.328,
        -3285.043,
        -8385.183,
        -8617.786,
        -8617.786,
        0.0,
    ]
    ssp245 = [
        39148.758,
        40647.530,
        43476.063,
        44252.900,
        43462.190,
        40196.485,
        35235.434,
        26838.373,
        16324.392,
        9682.859,
        9682.859 - 2582.1,
        0.0,
    ]
    ssp370 = [
        39148.758,
        44808.038,
        52847.359,
        58497.970,
        62904.059,
        66568.368,
        70041.979,
        73405.226,
        77799.049,
        82725.833,
        82725.833 - 22060.00,
        0.0,
    ]

    # HISTORICAL EMISSIONS
    hist_filename, x_hist = filemethods.get_emissions_filename()
    iy = np.where((x_hist >= 1850) & (x_hist < x_ssp[0]))[0]
    hist = pd.read_csv(data_dir + hist_filename).to_numpy()[iy, 0] * 3.664 * 1000.0
    x_hist = x_hist[iy]

    # hist = [
    #     27962.868,
    #     28811.292,
    #     29658.679,
    #     33413.896,
    #     36131.477,
    #     37951.325,
    #     39628.028,
    # ]
    # x_hist = [1990, 1995, 2000, 2005, 2010, 2012, 2014]

    # CONCATENATE AND INTERPOLATE
    ssp119_interp = (
        np.interp(
            x_interp,
            np.concatenate((x_hist, x_ssp), axis=0),
            np.concatenate((hist, ssp119), axis=0),
        )
        * 0.001
    )
    ssp126_interp = (
        np.interp(
            x_interp,
            np.concatenate((x_hist, x_ssp), axis=0),
            np.concatenate((hist, ssp126), axis=0),
        )
        * 0.001
    )
    ssp245_interp = (
        np.interp(
            x_interp,
            np.concatenate((x_hist, x_ssp), axis=0),
            np.concatenate((hist, ssp245), axis=0),
        )
        * 0.001
    )
    ssp370_interp = (
        np.interp(
            x_interp,
            np.concatenate((x_hist, x_ssp), axis=0),
            np.concatenate((hist, ssp370), axis=0),
        )
        * 0.001
    )

    i = np.where(ssp119_interp <= 0)[0]
    ssp119_yr = x_interp[i][0]

    i = np.where(ssp126_interp <= 0)[0]
    ssp126_yr = x_interp[i][0]

    # --------------------------------------------------------
    alpha = 0.8
    plot_emissions = False
    if plot_emissions:
        plt.figure(figsize=(7, 4))

        plt.axhline(y=0, color=colors["obs"], linewidth=1.0)

        i = np.where(x_interp <= 2015)[0]
        plt.plot(
            x_interp[i],
            ssp119_interp[i],
            linewidth=3,
            color=colors["obs"],
            alpha=alpha,
            label="Historical",
            zorder=1000,
        )
        i = np.where(x_interp >= 2015)[0]
        plt.plot(
            x_interp[i],
            ssp119_interp[i],
            linewidth=3,
            color=colors["ssp119"],
            alpha=alpha,
            label="SSP1-1.9",
            zorder=1000,
        )
        plt.plot(
            x_interp[i],
            ssp126_interp[i],
            linewidth=3,
            color=colors["ssp126"],
            alpha=alpha,
            label="SSP1-2.6",
        )
        plt.plot(
            x_interp[i],
            ssp245_interp[i],
            linewidth=3,
            color=colors["ssp245"],
            alpha=alpha,
            label="SSP2-4.5",
        )
        # plt.plot(
        #     x_interp[i],
        #     ssp370_interp[i],
        #     linewidth=3,
        #     color=colors["ssp370"],
        #     alpha=alpha,
        #     label="SSP3-7.0",
        # )

        plt.legend()

        plt.annotate(
            ssp119_yr,
            (ssp119_yr, 0),
            color=colors["ssp119"],
            xytext=(2045, -10.0),
            arrowprops=dict(
                arrowstyle="->", color=colors["ssp119"], connectionstyle="arc3"
            ),
        )

        plt.annotate(
            ssp126_yr,
            (ssp126_yr, 0),
            color=colors["ssp126"],
            xytext=(2082, 5),
            arrowprops=dict(
                arrowstyle="->", color=colors["ssp126"], connectionstyle="arc3"
            ),
        )

        plt.ylabel("Gt per year")
        plt.xlabel("year")

        # plots.format_spines(plt.gca())
        plt.xticks(np.arange(1850, 2250, 50), np.arange(1850, 2250, 50))
        plt.yticks(np.arange(-20, 100, 10), np.arange(-20, 100, 10))

        plt.xlim(1850, 2100)
        plt.ylim(-15.5, 85.5)

        plt.title("anthropogenic CO$_2$ emissions under Historical + SSPs")

        plots.savefig(figure_dir + "data_diagnostics/emissions")
        plt.close()
        raise ValueError

    # return emissions in Teratons
    if ssp == "ssp119":
        return ssp119_interp / 1000.0, x_interp
    elif ssp == "ssp126":
        return ssp126_interp / 1000.0, x_interp
    elif ssp == "ssp245":
        return ssp245_interp / 1000.0, x_interp
    elif ssp == "ssp370":
        return ssp370_interp / 1000.0, x_interp
    else:
        raise NotImplementedError()
