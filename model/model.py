"""Network modules for pytorch models.

Functions
---------
conv_couplet(in_channels, out_channels, act_fun, *args, **kwargs)
dense_lazy_couplet(out_features, act_fun, *args, **kwargs)
conv_block(in_channels, out_channels, act_fun, kernel_size)
dense_block(out_features, act_fun)


Classes
---------
RescaleLayer()
TorchModel(base.base_model.BaseModel)

"""

import torch
import numpy as np

from base.base_model import BaseModel

# https://github.com/FrancescoSaverioZuppichini/Pytorch-how-and-when-to-use-Module-Sequential-ModuleList-and-ModuleDict


def conv_couplet(in_channels, out_channels, act_fun, *args, **kwargs):
    return torch.nn.Sequential(
        torch.nn.Conv2d(in_channels, out_channels, *args, **kwargs),
        getattr(torch.nn, act_fun)(),
        torch.nn.MaxPool2d(kernel_size=(2, 2), ceil_mode=True),
    )


def dense_lazy_couplet(out_features, act_fun, *args, **kwargs):
    return torch.nn.Sequential(
        torch.nn.LazyLinear(out_features=out_features, bias=True),
        getattr(torch.nn, act_fun)(),
    )


def dense_couplet(in_features, out_features, act_fun, *args, **kwargs):
    return torch.nn.Sequential(
        torch.nn.Linear(in_features=in_features, out_features=out_features, bias=True),
        getattr(torch.nn, act_fun)(),
    )


def conv_block(in_channels, out_channels, act_fun, kernel_size):
    block = [
        conv_couplet(in_channels, out_channels, act_fun, kernel_size, padding="same")
        for in_channels, out_channels, act_fun, kernel_size in zip(
            [*in_channels],
            [*out_channels],
            [*act_fun],
            [*kernel_size],
        )
    ]
    return torch.nn.Sequential(*block)


def dense_block(out_features, act_fun, in_features=None):
    if in_features is None:
        block = [
            dense_lazy_couplet(out_channels, act_fun)
            for out_channels, act_fun in zip([*out_features], [*act_fun])
        ]
        return torch.nn.Sequential(*block)
    else:
        block = [
            dense_couplet(in_features, out_features, act_fun)
            for in_features, out_features, act_fun in zip(
                [*in_features], [*out_features], [*act_fun]
            )
        ]
        return torch.nn.Sequential(*block)


class RescaleLayer:
    def __init__(self, scale, offset):
        self.offset = offset
        self.scale = scale

    def __call__(self, x):
        x = torch.multiply(x, self.scale)
        x = torch.add(x, self.offset)
        return x


class TorchModel(BaseModel):
    def __init__(self, config, target_mean=None, target_std=None):
        super().__init__()

        self.config = config

        assert (
            len(self.config["cnn_act"])
            == len(self.config["kernel_size"])
            == len(self.config["filters"])
        )
        assert len(self.config["hiddens_block"]) == len(
            self.config["hiddens_block_act"]
        )

        if target_mean is None:
            self.target_mean = torch.tensor(0.0)
        else:
            self.target_mean = torch.tensor(target_mean)

        if target_std is None:
            self.target_std = torch.tensor(1.0)
        else:
            self.target_std = torch.tensor(target_std)

        # Longitude padding
        self.pad_lons = torch.nn.CircularPad2d(config["circular_padding"])

        # CNN block
        self.conv_block = conv_block(
            [config["n_inputchannel"], *config["filters"][:-1]],
            [*config["filters"]],
            [*config["cnn_act"]],
            [*config["kernel_size"]],
        )

        # Flat layer
        self.flat = torch.nn.Flatten(start_dim=1)

        # Dense blocks
        self.denseblock_mu = dense_block(
            config["hiddens_block"],
            config["hiddens_block_act"],
            in_features=config["hiddens_block_in"],
        )
        self.denseblock_sigma = dense_block(
            config["hiddens_block"],
            config["hiddens_block_act"],
            in_features=config["hiddens_block_in"],
        )
        self.denseblock_gamma = dense_block(
            config["hiddens_block"],
            config["hiddens_block_act"],
            in_features=config["hiddens_block_in"],
        )
        self.denseblock_tau = dense_block(
            config["hiddens_block"],
            config["hiddens_block_act"],
            in_features=config["hiddens_block_in"],
        )

        # Final dense layer
        self.finaldense_mu = dense_couplet(
            out_features=config["hiddens_final"],
            act_fun=config["hiddens_final_act"],
            in_features=config["hiddens_final_in"],
        )
        self.finaldense_sigma = dense_couplet(
            out_features=config["hiddens_final"],
            act_fun=config["hiddens_final_act"],
            in_features=config["hiddens_final_in"],
        )
        self.finaldense_gamma = dense_couplet(
            out_features=config["hiddens_final"],
            act_fun=config["hiddens_final_act"],
            in_features=config["hiddens_final_in"],
        )
        self.finaldense_tau = dense_couplet(
            out_features=config["hiddens_final"],
            act_fun=config["hiddens_final_act"],
            in_features=config["hiddens_final_in"],
        )

        # Rescaling layers
        self.rescale_mu = RescaleLayer(self.target_std, self.target_mean)
        self.rescale_sigma = RescaleLayer(torch.tensor(1.0), torch.log(self.target_std))
        if "gamma" in config.get("freeze_id", []):
            self.rescale_gamma = RescaleLayer(torch.tensor(0.0), torch.tensor(0.0))
        else:
            self.rescale_gamma = RescaleLayer(torch.tensor(1.0), torch.tensor(0.0))

        if "tau" in config.get("freeze_id", []):
            self.rescale_tau = RescaleLayer(torch.tensor(0.0), torch.tensor(0.0))
        else:
            self.rescale_tau = RescaleLayer(torch.tensor(1.0), torch.tensor(0.0))

        # Output layers
        self.output_mu = torch.nn.Linear(
            in_features=config["hiddens_final"], out_features=1, bias=True
        )
        self.output_sigma = torch.nn.Linear(
            in_features=config["hiddens_final"], out_features=1, bias=True
        )
        self.output_gamma = torch.nn.Linear(
            in_features=config["hiddens_final"], out_features=1, bias=True
        )
        self.output_tau = torch.nn.Linear(
            in_features=config["hiddens_final"], out_features=1, bias=True
        )

    def forward(self, x, x_unit):

        x = self.pad_lons(x)
        x = self.conv_block(x)
        x = self.flat(x)

        # build mu_layers
        x_mu = torch.cat((x, x_unit[:, None]), dim=-1)
        x_mu = self.denseblock_mu(x_mu)
        x_mu = torch.cat((x_mu, x_unit[:, None]), dim=-1)
        x_mu = self.finaldense_mu(x_mu)
        mu_out = self.output_mu(x_mu)

        # build sigma_layers
        x_sigma = torch.cat((x, x_unit[:, None]), dim=-1)
        x_sigma = self.denseblock_sigma(x_sigma)
        x_sigma = torch.cat((x_sigma, x_unit[:, None]), dim=-1)
        x_sigma = self.finaldense_sigma(x_sigma)
        sigma_out = self.output_sigma(x_sigma)

        # build gamma_layers
        x_gamma = torch.cat((x, x_unit[:, None]), dim=-1)
        x_gamma = self.denseblock_gamma(x_gamma)
        x_gamma = torch.cat((x_gamma, x_unit[:, None]), dim=-1)
        x_gamma = self.finaldense_gamma(x_gamma)
        gamma_out = self.output_gamma(x_gamma)

        # build tau_layers
        x_tau = torch.cat((x, x_unit[:, None]), dim=-1)
        x_tau = self.denseblock_tau(x_tau)
        x_tau = torch.cat((x_tau, x_unit[:, None]), dim=-1)
        x_tau = self.finaldense_tau(x_tau)
        tau_out = self.output_tau(x_tau)

        # rescaling layers
        mu_out = self.rescale_mu(mu_out)

        sigma_out = self.rescale_sigma(sigma_out)
        sigma_out = torch.exp(sigma_out)

        gamma_out = self.rescale_gamma(gamma_out)

        tau_out = self.rescale_tau(tau_out)
        tau_out = torch.exp(tau_out)

        # final output, concatenate parameters together
        x = torch.cat((mu_out, sigma_out, gamma_out, tau_out), dim=-1)

        return x

    def predict(self, dataset=None, dataloader=None, batch_size=128, device="cpu"):

        if (dataset is None) & (dataloader is None):
            raise ValueError("both dataset and dataloader cannot be done.")

        if (dataset is not None) & (dataloader is not None):
            raise ValueError("dataset and dataloader cannot both be defined. choose one.")

        if dataset is not None:
            dataloader = torch.utils.data.DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=False,
                drop_last=False,
            )

        self.to(device)
        self.eval()
        with torch.inference_mode():
            output = None
            for batch_idx, (data, target) in enumerate(dataloader):
                input, input_unit, target = (
                    data[0].to(device),
                    data[1].to(device),
                    target.to(device),
                )

                out = self(input, input_unit).to("cpu").numpy()
                if output is None:
                    output = out
                else:
                    output = np.concatenate((output, out), axis=0)

        return output
