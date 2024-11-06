"""Utility classes and functions.

Functions
---------
prepare_device(device="gpu")
save_torch_model(model, filename)
load_torch_model(model, filename)
get_config(exp_name)
get_model_name(expname, seed)
cubicFunc(x, intercept, slope_1, slope_2, slope_3)

Classes
---------
MetricTracker()

"""

import json
import torch
import pandas as pd
import numpy as np


def prepare_device(device="gpu"):
    """
    setup GPU device if available. get gpu device indices which are used for DataParallel
    """
    if device == "gpu":
        if torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            print("Warning: MPS device not found." "Training will be performed on CPU.")
            device = torch.device("cpu")
    elif device == "cpu":
        device = torch.device("cpu")
    else:
        raise NotImplementedError

    return device


def save_torch_model(model, filename):
    if filename[-3:] != ".pt":
        filename = filename + ".pt"

    torch.save(model.state_dict(), filename)


def load_torch_model(model, filename):
    model.load_state_dict(torch.load(filename))
    model.eval()
    return model


def get_config(exp_name):

    basename = "exp"

    with open("config/config_" + exp_name[len(basename) :] + ".json") as f:
        config = json.load(f)

    assert config["expname"] == basename + exp_name[len(basename) :], "Exp_Name must be equal to config[exp_name]"

    # add additional attributes for easier use later
    config["datamaker"]["fig_dpi"] = config["fig_dpi"]
    config["datamaker"]["data_dir"] = config["data_dir"]

    return config


def get_model_name(expname, seed):
    model_name = expname + "_seed" + str(seed)

    return model_name


def cubicFunc(x, intercept, slope_1, slope_2, slope_3):
    return intercept + slope_1 * x + slope_2 * (x**2) + slope_3 * (x**3)


class MetricTracker:
    """Could have written this as a subclass of dict() itself, but instead it can now
    hold other attributes if desired.
    """

    def __init__(self, *keys):

        self.history = dict()
        for k in keys:
            self.history[k] = []
        self.reset()

    def reset(self):
        for key in self.history:
            self.history[key] = []

    def update(self, key, value):
        if key in self.history:
            self.history[key].append(value)

    def result(self):
        for key in self.history:
            self.history[key] = np.nanmean(self.history[key])

    def print(self, idx=None):
        for key in self.history.keys():
            if idx is None:
                print(f"  {key} = {self.history[key]:.5f}")
            else:
                print(f"  {key} = {self.history[key][idx]:.5f}")
