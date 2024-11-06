"""Data maker modules.

Functions
---------


Classes
---------
ClimateData()

"""

import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import copy
import numpy as np

import utils
import visuals.plots as plots
import datamaker.emissions as emissions
import datamaker.regions as regions
import datamaker.filemethods as filemethods
from datamaker.sample_vault import SampleDict


class ClimateData:
    """
    Custom dataset for climate data and processing.
    """

    def __init__(
        self, config, expname, seed, data_dir, figure_dir, fetch=True, verbose=False
    ):

        self.config = config
        self.expname = expname
        self.seed = seed
        self.data_dir = data_dir
        self.figure_dir = figure_dir
        self.verbose = verbose

        if self.config.get("fixed_seed", None) is not None:
            self.rng = np.random.default_rng(self.config.get("fixed_seed") + 42)
        else:
            self.rng = np.random.default_rng(self.seed + 42)

        if fetch:
            self.fetch_data()

    def fetch_obs(self, ssp, verbose=None):
        if verbose is not None:
            self.verbose = verbose

        self.d_obs = SampleDict()

        self._create_obs(ssp)

        if self.verbose:
            self.d_obs.summary()

    def fetch_data(self, verbose=None):
        if verbose is not None:
            self.verbose = verbose

        self.d_train = SampleDict()
        self.d_val = SampleDict()
        self.d_test = SampleDict()

        self._get_members()
        self._create_data()

        if self.verbose:
            self.d_train.summary()
            self.d_val.summary()
            self.d_test.summary()

    def _create_obs(self, ssp):
        self._ssp = ssp

        filename = filemethods.get_observations_filename(
            source=self.config["obs_source"], verbose=self.verbose
        )
        da = filemethods.get_netcdf_da(self.data_dir + filename)
        da = da.expand_dims(dim="member", axis=0)

        f_dict = self._process_data(
            da,
            members=(0,),
            gcm_name=self.config["obs_source"],
        )
        self.d_obs.concat(f_dict)

        # reshape the data into samples
        self.d_obs.reshape()

        # Re-calibrate obs values according to OBS data in 2023 and set max_temp to nan since unknown
        if self.config["obs_source"] == "BEST":
            # https://berkeleyearth.org/global-temperature-report-for-2023/
            self.d_obs.calibrate_temp(
                key="current_temp", calibrate_year=2023, calibrate_value=1.54
            )
        elif self.config["obs_source"] == "GISTEMP":
            self.d_obs.calibrate_temp(
                key="current_temp", calibrate_year=2023, calibrate_value=1.37
            )
        elif self.config["obs_source"] == "ERA5":
            self.d_obs.calibrate_temp(
                key="current_temp", calibrate_year=2023, calibrate_value=1.48
            )
        else:
            raise NotImplementedError
        self.d_obs["max_temp"] = self.d_obs["max_temp"] * np.nan

        # Fill nans with zeros
        if self.config["anomalies"]:
            self.d_obs["x"] = np.nan_to_num(self.d_obs["x"], 0.0)

        # add latitude and longitude
        self.lat = da.lat.values
        self.lon = da.lon.values

        # cleanup
        del self._ssp

    def _create_data(self):

        # get the SSP list
        for isub, sub in enumerate(self.config["gcmsub"]):
            for ssp in self.config["ssp_list"]:
                self._ssp = ssp
                filenames = filemethods.get_cmip_filenames(ssp, sub)

                for f in filenames:
                    if self.verbose:
                        print(f)

                    self._get_members()
                    da = filemethods.get_netcdf_da(self.data_dir + f)
                    gcm_name = filemethods.get_gcm_name(f, ssp)

                    # get processed X and Y data
                    # process the data, i.e. compute anomalies, subtract the mean, etc.
                    f_dict_train = self._process_data(
                        da, members=self.train_members[isub], gcm_name=gcm_name
                    )
                    f_dict_val = self._process_data(
                        da, members=self.val_members[isub], gcm_name=gcm_name
                    )
                    f_dict_test = self._process_data(
                        da, members=self.test_members[isub], gcm_name=gcm_name
                    )

                    # select training years
                    f_dict_train = self._select_training_years(f_dict_train)
                    f_dict_val = self._select_training_years(f_dict_val)
                    f_dict_test = self._select_training_years(f_dict_test)

                    # # concatenate with the rest of the data
                    self.d_train.concat(f_dict_train)
                    self.d_val.concat(f_dict_val)
                    self.d_test.concat(f_dict_test)

        # reshape the data into samples
        self.d_train.reshape()
        self.d_val.reshape()
        self.d_test.reshape()

        # clean-up the train/val/test data to remove np.nan samples
        self.d_train.del_nans()
        self.d_val.del_nans()
        self.d_test.del_nans()

        # add latitude and longitude
        self.lat = da.lat.values
        self.lon = da.lon.values

        # cleanup
        del self._ssp

    def _process_data(self, da, members=None, gcm_name=None):

        # --------------
        # CREATE THE FILE-DATA DICTIONARY
        f_dict = SampleDict()

        # --------------
        # CHECK IF EMPTY MEMBERS, if so, RETURN
        if len(members) == 0:
            return f_dict

        # --------------
        # SELECT MAXIMUM YEAR RANGE NEEDED
        da = da.sel(
            time=slice(
                str(self.config["complete_yr_bounds"][0]),
                str(self.config["complete_yr_bounds"][1]),
            )
        )

        # --------------
        # SELECT THE MEMBERS
        if members is None:
            da_members = da
        else:
            if not isinstance(members, np.ndarray):
                members = np.asarray(members)
            da_members = da[members, :, :, :]

        # --------------
        # ADD CHANNEL DIMENSION
        da_members = da_members.expand_dims(dim={"channel": 1}, axis=-1)

        # --------------
        # GET THE SCALAR DATA (e.g. labels, years, etc.)
        f_dict = self._get_scalar_data(
            f_dict, da, members, figname=gcm_name, plot=False
        )

        # --------------
        # GET THE X-DATA
        f_dict = self._get_gridded_data(f_dict, da_members)

        # --------------
        # GET EMISSIONS
        _, cum_emiss_left, _ = emissions.create_emissions(
            self._ssp,
            f_dict["year"][0, :],
            self.data_dir,
            self.figure_dir,
            self.config["emissions_endyear"],
            plot=False,
        )
        f_dict["emissions_left"] = np.tile(cum_emiss_left, (f_dict["y"].shape[0], 1))

        # --------------
        # INSERT META DATA
        f_dict["gcm_name"] = np.tile(gcm_name, f_dict["y"].shape)
        f_dict["ssp"] = np.tile(self._ssp, f_dict["y"].shape)
        f_dict["member"] = np.tile(members[:, np.newaxis], f_dict["y"].shape[1])

        return f_dict

    def _get_gridded_data(self, f_dict, da):
        if self.config["anomalies"]:
            da_anomalies = da - da.sel(
                time=slice(
                    str(self.config["anomaly_yr_bounds"][0]),
                    str(self.config["anomaly_yr_bounds"][1]),
                )
            ).mean("time")
        elif self.config["anomalies"] == "baseline":
            if self.verbose:
                print("computing anomalies relative to baseline' ...")
            da_anomalies = da - da.sel(
                time=slice(
                    str(self.config["baseline_yr_bounds"][0]),
                    str(self.config["baseline_yr_bounds"][1]),
                )
            ).mean("time")

        elif not self.config["anomalies"]:
            print("not computing any anomalies...")
            pass
        else:
            raise NotImplementedError()

        if self.config["remove_map_mean"] == "raw":
            da_anomalies = da_anomalies - da_anomalies.mean(("lon", "lat"))
        elif self.config["remove_map_mean"] == "weighted":
            weights = np.cos(np.deg2rad(da_anomalies.lat))
            weights.name = "weights"
            da_anomalies_weighted = da_anomalies.weighted(weights)
            da_anomalies = da_anomalies - da_anomalies_weighted.mean(("lon", "lat"))

        f_dict["x"] = da_anomalies.values

        return f_dict

    def _get_scalar_data(self, f_dict, da, members, figname="", plot=False):
        data_output, _, _ = regions.extract_region(
            da, region=self.config["target_region"], dir=self.config["data_dir"]
        )
        global_avg_mean = regions.compute_global_mean(data_output.mean(axis=0))
        global_avg_mem = regions.compute_global_mean(data_output)

        # compute the target year
        baseline_mean = global_avg_mem.sel(
            time=slice(
                str(self.config["baseline_yr_bounds"][0]),
                str(self.config["baseline_yr_bounds"][1]),
            )
        ).mean("time")

        if self.config["label_forced"]:
            if self.config["label_ensmean"]:

                # smooth the global mean timeseries
                anomalies = global_avg_mean - baseline_mean
                years = anomalies["time.year"].values
                iyrs = np.where(years >= self.config["fit_start_year"])[0]
                fit, _ = curve_fit(utils.cubicFunc, years[iyrs], anomalies.values[iyrs])
                anomalies = utils.cubicFunc(years[iyrs], *fit)

                # compute the maximum temperature reached
                max_temp = np.max(anomalies, axis=-1)
                imax = np.argmax(anomalies, axis=-1)
                year_reached = global_avg_mean["time.year"].values[iyrs[imax]]

                if not isinstance(max_temp, np.ndarray):
                    max_temp = np.ones((da.shape[0])) * max_temp
                    year_reached = np.asarray(
                        np.ones((da.shape[0])) * year_reached, dtype="int"
                    )
            else:
                anomalies = global_avg_mem - baseline_mean
                years = anomalies["time.year"].values
                anomalies = anomalies.values
                iyrs = np.where(years >= self.config["fit_start_year"])[0]
                for mem in range(anomalies.shape[0]):
                    fit, _ = curve_fit(
                        utils.cubicFunc, years[iyrs], anomalies[mem, iyrs, ...]
                    )
                    anomalies[mem, iyrs, ...] = utils.cubicFunc(years[iyrs], *fit)
                anomalies = anomalies[:, iyrs, ...]
                # anomalies[mem, ...] = utils.cubicFunc(years[iyrs], *fit)

                # compute the maximum temperature reached
                max_temp = np.max(anomalies, axis=-1)
                imax = np.argmax(anomalies, axis=-1)
                year_reached = global_avg_mem["time.year"].values[iyrs[imax]]

        else:
            # grab each member separately
            anomalies = global_avg_mem - baseline_mean
            years = anomalies["time.year"].values

            iyrs = np.arange(0, len(years))
            anomalies = anomalies.values

            # compute the maximum temperature reached
            max_temp = np.max(anomalies, axis=-1)
            imax = np.argmax(anomalies, axis=-1)
            year_reached = global_avg_mem["time.year"].values[iyrs[imax]]

        if self.verbose:
            print(f"  {max_temp.round(2) = }\n" f"  {year_reached = }")

            # --------------------
            # plot the results
            plots.plot_label_definition(
                year_reached,
                max_temp,
                global_avg_mem,
                global_avg_mean,
                baseline_mean,
                anomalies,
                iyrs,
                self.config,
            )
            plt.title(figname + ": " + self._ssp, fontsize=8)
            plots.savefig(
                self.figure_dir
                + "data_diagnostics/"
                + self.expname
                + "_"
                + figname
                + "_label_definition",
                fig_format=(".png", ".pdf"),
                dpi=self.config["fig_dpi"],
            )
            plt.close()

        # --------------------
        # define the labels
        current_temp = global_avg_mem.values - baseline_mean.values[:, np.newaxis]
        labels = max_temp[:, np.newaxis] - (
            global_avg_mem.values - baseline_mean.values[:, np.newaxis]
        )

        # remove any temperature deviations after the max is reached
        if self.config["del_after_max_samples"]:
            for ens in np.arange(0, len(year_reached)):
                labels[ens, :] = np.where(
                    global_avg_mean["time.year"].values > year_reached[ens],
                    np.nan,
                    labels[ens, :],
                )

        # ----------------------------------------------------------
        # ONLY GRAB THE MEMBERS THAT WE WANT
        if members is not None:
            labels = labels[members, :]
            current_temp = current_temp[members, :]
            max_temp = max_temp[members]
            year_reached = year_reached[members]

        assert labels.shape == current_temp.shape
        assert labels.shape[0] == max_temp.shape[0]

        # ----------------------------------------------------------
        # RESHAPE DATA INTO SAMPLES AND STORE IN RETURNED DICTIONARY
        f_dict["y"] = labels
        f_dict["current_temp"] = current_temp
        f_dict["max_temp"] = np.tile(max_temp[:, np.newaxis], f_dict["y"].shape[1])
        f_dict["year_reached"] = np.tile(
            year_reached[:, np.newaxis], f_dict["y"].shape[1]
        )
        f_dict["year"] = np.tile(da["time.year"].values, (f_dict["y"].shape[0], 1))

        assert np.sum(np.diff(f_dict["max_temp"], axis=1)) == 0

        if self.verbose:
            print(f"  {f_dict['y'].shape = }")

        return f_dict

    def _get_members(self):

        self.train_members = []
        self.val_members = []
        self.test_members = []

        for splitvec in self.config["n_train_val_test"]:

            # get number of members or fraction of members
            n_train = splitvec[0]
            n_val = splitvec[1]
            n_test = splitvec[2]

            # if given as a fraction rather than actual numbers
            # then data is likely single_member
            if n_train < 1:
                assert np.sum(splitvec) == 1

                rv = self.rng.uniform(low=0, high=1, size=1)
                if rv <= n_train:
                    train_members, val_members, test_members = [0], [], []
                elif (rv > n_train) & (rv < n_train + n_val):
                    train_members, val_members, test_members = [], [0], []
                else:
                    train_members, val_members, test_members = [], [], [0]

            else:
                # data is multi-member data
                # choose members to train based on fixed seed if requested
                if self.config.get("fixed_seed", None) is not None:
                    rng_cmip = np.random.default_rng(self.config.get("fixed_seed"))
                else:
                    rng_cmip = np.random.default_rng(self.seed)

                if ("single_member" in self.config["gcmsub"]) & (
                    "multi_member" in self.config["gcmsub"]
                ):
                    # do not include first member since it is already
                    # used in the multi-member data grab
                    all_members = np.arange(1, n_train + n_val + n_test + 1)
                else:
                    all_members = np.arange(0, n_train + n_val + n_test)

                train_members = rng_cmip.choice(
                    all_members, size=n_train, replace=False
                ).tolist()
                val_members = rng_cmip.choice(
                    np.setdiff1d(all_members, train_members), size=n_val, replace=False
                ).tolist()
                test_members = rng_cmip.choice(
                    np.setdiff1d(all_members, np.append(train_members[:], val_members)),
                    size=n_test,
                    replace=False,
                ).tolist()
            self.train_members.append(train_members)
            self.val_members.append(val_members)
            self.test_members.append(test_members)

        if self.verbose:
            print(
                f"Member for train/val/test split: {self.train_members} / {self.val_members} / {self.test_members}"
            )

    def _select_training_years(self, f_dict):

        if len(f_dict["y"]) == 0:
            return f_dict

        # only train on certain samples
        iyears = copy.deepcopy(
            np.where(
                (f_dict["year"][0, :] >= self.config["training_yr_bounds"][0])
                & (f_dict["year"][0, :] <= self.config["training_yr_bounds"][1])
            )[0]
        )
        f_dict.subsample(idx=iyears, axis=1)

        return f_dict
