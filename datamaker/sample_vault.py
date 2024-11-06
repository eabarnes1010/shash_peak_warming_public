"""Sample vault modules for storing data samples in dictionaries.

Classes
---------
SampleDict(dict)

"""

import numpy as np
import copy


class SampleDict(dict):
    def __init__(self, *arg, **kw):
        super(SampleDict, self).__init__(*arg, **kw)

        self.__setitem__("x", [])
        self.__setitem__("y", [])
        self.__setitem__("year", [])
        self.__setitem__("current_temp", [])
        self.__setitem__("max_temp", [])
        self.__setitem__("year_reached", [])
        self.__setitem__("member", [])
        self.__setitem__("emissions_left", [])
        self.__setitem__("gcm_name", [])
        self.__setitem__("ssp", [])

    def summary(self):
        for key in self:
            print(f"data[{key}].shape = {self[key].shape}")
        print("\n")

    def reset(self):
        for key in self:
            self[key] = []

    def reshape(self):
        for key in self:
            if len(self[key]) == 0:
                continue

            if key == "x":
                self["x"] = self["x"].reshape(
                    (
                        self["x"].shape[0] * self["x"].shape[1],
                        self["x"].shape[2],
                        self["x"].shape[3],
                        self["x"].shape[4],
                    )
                )
            else:
                self[key] = self[key].reshape((self[key].shape[0] * self[key].shape[1],))

    def del_nans(self):

        if len(self["y"]) == 0:
            return

        assert len(self["y"].shape) == 1
        inot_nan = np.where(~np.isnan(self["y"]))[0]

        for key in self:
            if len(self[key]) == 0:
                continue
            self[key] = self[key][inot_nan]

    def subsample(self, idx, axis=0, use_copy=False):
        if use_copy:
            d = copy.deepcopy(self)
            if axis == 0:
                for key in d:
                    d[key] = d[key][idx, ...]
            elif axis == 1:
                for key in d:
                    d[key] = d[key][:, idx, ...]
            else:
                raise NotImplementedError
            return d

        elif not use_copy:
            if axis == 0:
                for key in self:
                    self[key] = self[key][idx, ...]
            elif axis == 1:
                for key in self:
                    self[key] = self[key][:, idx, ...]
            else:
                raise NotImplementedError

        else:
            raise NotImplementedError

    def concat(self, f_dict):
        for key in self:
            if len(self[key]) == 0:
                self[key] = f_dict[key]
            elif len(f_dict[key]) == 0:
                pass
            else:
                self[key] = np.concatenate((self[key], f_dict[key]), axis=0)

    def calibrate_temp(self, key, calibrate_year, calibrate_value):
        idx = np.where(self["year"] == calibrate_year)[0]
        assert len(idx) == 1
        self[key] = self[key] - self[key][idx] + calibrate_value
