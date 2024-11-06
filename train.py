"""
This script trains a model for peak temperature prediction using climate data. It loads the necessary modules and libraries, sets up the data, builds and trains the model, computes metrics, and saves the model and metrics.

Usage:
    python train.py <expname>

Arguments:
    expname (str): The experiment name to specify the config file, e.g. exp101

Example:
    python train.py exp101
"""

import sys
import xarray as xr
import torch
import random
import matplotlib.pyplot as plt
import numpy as np
import torchinfo
import importlib as imp
import pandas as pd
import warnings
import argparse
import os.path

from datamaker.data_generator import ClimateData
from trainer.trainer import Trainer
from model.model import TorchModel
from utils import utils
import model.loss as module_loss
import model.metric as module_metric
import visuals.plots as plots
from shash.shash_torch import Shash
import datamaker.data_loader as data_loader

warnings.filterwarnings("ignore")
torch.set_warn_always(False)

# print(f"python version = {sys.version}")
# print(f"numpy version = {np.__version__}")
# print(f"xarray version = {xr.__version__}")
# print(f"pytorch version = {torch.__version__}")

# --------------------------------------------------------
OVERWRITE = False

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "expname", help="experiment name to specify the config file, e.g. exp101"
    )
    args = parser.parse_args()
    config = utils.get_config(args.expname)

    # Loop through random seeds
    for seed in config["seed_list"]:

        # make model initialization deterministic
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True

        model_name = utils.get_model_name(config["expname"], seed)
        if (os.path.isfile(config["model_dir"] + model_name + ".pt")) & (not OVERWRITE):
            continue
        print("___________________")
        print(model_name)

        # Get the Data
        print("___________________")
        print("Get the data.")
        data = ClimateData(
            config["datamaker"],
            expname=config["expname"],
            seed=seed,
            data_dir=config["data_dir"],
            figure_dir=config["figure_dir"],
            verbose=True,
        )

        trainset = data_loader.CustomData(data.d_train)
        valset = data_loader.CustomData(data.d_val)
        testset = data_loader.CustomData(data.d_test)

        train_loader = torch.utils.data.DataLoader(
            trainset,
            batch_size=config["datamaker"]["batch_size"],
            shuffle=True,
            drop_last=False,
        )
        val_loader = torch.utils.data.DataLoader(
            valset,
            batch_size=config["datamaker"]["batch_size"],
            shuffle=False,
            drop_last=False,
        )

        # Setup the Model
        print("___________________")
        print("Building and training the model.")

        model = TorchModel(
            config=config["arch"],
            target_mean=trainset.target.mean(axis=0),
            target_std=trainset.target.std(axis=0),
        )
        if config["arch"].get("freeze_id", -1) != -1:
            for freeze_id in config["arch"]["freeze_id"]:
                print(f"..freezing layers containing --> {freeze_id}")
                model.freeze_layers(freeze_id=freeze_id)

        optimizer = getattr(torch.optim, config["optimizer"]["type"])(
            model.parameters(), **config["optimizer"]["args"]
        )
        criterion = getattr(module_loss, config["criterion"])()
        scheduler = getattr(torch.optim.lr_scheduler, config["scheduler"]["type"])(
            optimizer, **config["scheduler"]["args"]
        )

        metric_funcs = [getattr(module_metric, met) for met in config["metrics"]]

        # Build the trainer
        device = utils.prepare_device(config["device"])
        trainer = Trainer(
            model,
            criterion,
            metric_funcs,
            optimizer,
            scheduler,
            max_epochs=config["trainer"]["max_epochs"],
            data_loader=train_loader,
            validation_data_loader=val_loader,
            device=device,
            config=config,
        )

        # Visualize the model
        torchinfo.summary(
            model,
            [
                trainset.input[: config["datamaker"]["batch_size"]].shape,
                trainset.input_unit[: config["datamaker"]["batch_size"]].shape,
            ],
            verbose=0,
            col_names=("input_size", "output_size", "num_params"),
        )

        # Train the Model
        model.to(device)
        trainer.fit()
        model.eval()

        # Save the Pytorch Model
        utils.save_torch_model(model, config["model_dir"] + model_name + ".pt")

        # Compute metrics and make visualizations
        # Make and save cmip predictions for train/val/test
        print("___________________")
        print("Computing metrics and assessing the model predictions.")

        # Make predictions for train/val/test
        with torch.inference_mode():
            # output_train = model.predict(dataset=trainset, batch_size=128, device=device)
            output_val = model.predict(dataset=valset, batch_size=128, device=device)
            output_test = model.predict(dataset=testset, batch_size=128, device=device)

        # ----------------------------------------
        # Compute and save the final metrics
        error_val = module_metric.custom_mae(output_val, data.d_val["y"])
        error_test = module_metric.custom_mae(output_test, data.d_test["y"])

        _, _, d_val, _ = module_metric.pit_d(output_val, data.d_val["y"])
        _, _, d_test, _ = module_metric.pit_d(output_test, data.d_test["y"])
        _, _, d_valtest, _ = module_metric.pit_d(
            np.append(output_val, output_test, axis=0),
            np.append(data.d_val["y"], data.d_test["y"], axis=0),
        )

        loss_val = criterion(output_val, data.d_val["y"][None, :]).flatten().numpy()
        loss_test = criterion(output_test, data.d_test["y"][None, :]).flatten().numpy()

        # fill and save the metrics dictionary
        d = {}
        d["expname"] = config["expname"]
        d["rng_seed"] = seed
        d["target_mean"] = model.target_mean.numpy()
        d["target_std"] = model.target_std.numpy()
        d["error_val"] = error_val
        d["error_test"] = error_test
        d["loss_val"] = loss_val
        d["loss_test"] = loss_test
        d["d_val"] = d_val
        d["d_test"] = d_test
        d["d_valtest"] = d_valtest

        df = pd.DataFrame(d, index=[0]).reset_index(drop=True)
        df.to_pickle(config["output_dir"] + model_name + "_metrics.pickle")

        # ----------------------------------------
        # create and save diagnostics plots
        plots.plot_metrics_panels(trainer, config)
        plots.savefig(
            config["figure_dir"]
            + "model_diagnostics/"
            + model_name
            + "_metrics_diagnostic",
            fig_format=(".png",),
            dpi=config["fig_dpi"],
        )
        plt.close()

        plots.plot_one_to_one_diagnostic(
            output_val,
            output_test,
            data.d_val["y"],
            data.d_test["y"],
            data.d_test["year"],
        )
        plots.savefig(
            config["figure_dir"]
            + "model_diagnostics/"
            + model_name
            + "_one_to_one_diagnostic",
            fig_format=(".png",),
            dpi=config["fig_dpi"],
        )
        plt.close()

        plots.plot_pits(output_val, data.d_val["y"])
        plots.savefig(
            config["figure_dir"] + "model_diagnostics/" + model_name + "_pit",
            fig_format=(".png",),
            dpi=config["fig_dpi"],
        )
        plt.close()

        plots.plot_predicted_max(
            data.d_test, Shash(output_test).median().numpy(), config["expname"]
        )
        plots.savefig(
            config["figure_dir"]
            + "model_diagnostics/"
            + model_name
            + "_predicted_maxtemp",
            fig_format=(".png",),
            dpi=config["fig_dpi"],
        )
        plt.close()

        print("Completed " + model_name)
