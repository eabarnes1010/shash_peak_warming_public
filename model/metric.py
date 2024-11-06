"""Metrics for training and evaluation.

Functions
---------
custom_mae(output, target)
iqr_capture(output, target)
sign_test(output, target)
pit(output, target)

"""

import torch
from shash.shash_torch import Shash
import numpy as np


def custom_mae(output, target):
    """Compute the prediction mean absolute error.
    The "predicted value" is the median of the conditional distribution.

    """
    with torch.no_grad():

        assert len(output[:, 0]) == len(target)

        dist = Shash(output)
        return torch.mean(torch.abs(dist.median() - target)).item()


def iqr_capture(output, target):
    """Compute the fraction of true values between the 25 and 75 percentiles
    (i.e. the interquartile range).

    """
    with torch.no_grad():
        assert len(output[:, 0]) == len(target)

        dist = Shash(output)
        lower = dist.quantile(torch.tensor(0.25))
        upper = dist.quantile(torch.tensor(0.75))
        count = torch.sum(
            torch.logical_and(torch.greater(target, lower), torch.less(target, upper))
        ).item()

        return count / len(target)


def sign_test(output, target):
    """Compute the fraction of true values above the median."""
    with torch.no_grad():
        assert len(output[:, 0]) == len(target)

        dist = Shash(output)
        median = dist.quantile(torch.tensor(0.50))
        count = torch.sum(torch.greater(target, median)).item()

        return count / len(target)


def pit_d(output, target):
    """Compute the PIT (Probability Integral Transform) histogram."""
    bins = np.linspace(0, 1, 11)

    dist = Shash(output)
    F = dist.cdf(target)
    pit_hist = np.histogram(
        F,
        bins,
        weights=np.ones_like(F) / float(len(F)),
    )

    B = len(pit_hist[0])
    D = np.sqrt(1 / B * np.sum((pit_hist[0] - 1 / B) ** 2))
    EDp = np.sqrt((1.0 - 1 / B) / (target.shape[0] * B))

    return bins, pit_hist, D, EDp
