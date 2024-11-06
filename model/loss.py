"""Losses for training.

Classes
---------
ShashNLL(torch.nn.Module)
GaussianNLL(torch.nn.Module)

"""

import torch
import pandas as pd
import numpy as np
from shash.shash_torch import Shash


class ShashNLL(torch.nn.Module):
    """
    Negative log likelihood loss for a SHASH distribution.
    """

    def __init__(self):
        super(ShashNLL, self).__init__()

        self.epsilon = 1.0e-07

    def forward(self, output, target):

        dist = Shash(output)
        loss = -dist.log_prob(target)

        # TODO: might want to keep this setup the entire time.
        # to prevent huge initial losses and improve stability
        # loss = -torch.log(dist.prob(target) + self.epsilon)

        return loss.mean()


class GaussianNLL(torch.nn.Module):
    """
    Negative log likelihood loss for a Normal distribution.
    """

    def __init__(self):
        super(GaussianNLL, self).__init__()

        self.epsilon = 1.0e-07

    def forward(self, output, target):
        loc = output[:, 0]
        scale = output[:, 1]

        loss = -torch.distributions.normal.Normal(loc=loc, scale=scale).log_prob(target)

        # to prevent huge initial losses and improve stability
        # loss = -torch.log(
        #     torch.exp(torch.distributions.normal.Normal(loc=loc, scale=scale).log_prob(target))
        #     + self.epsilon
        # )

        return loss.mean()
