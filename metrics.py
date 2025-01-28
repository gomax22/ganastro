import torch
import numpy as np
import itertools
import logging
import time

import numpy as np
import torch

from sbibm.third_party.igms.main import ExpQuadKernel as tp_ExpQuadKernel
from sbibm.third_party.igms.main import mmd2_unbiased as tp_mmd2_unbiased
from sbibm.third_party.torch_two_sample.main import MMDStatistic as tp_MMDStatistic
from sbibm.utils.torch import get_default_device

log = logging.getLogger(__name__)


def mmd(
    X: torch.Tensor,
    Y: torch.Tensor,
    implementation: str = "tp_sutherland",
    z_score: bool = False,
    bandwidth: str = "X",
) -> torch.Tensor:
    """Estimate MMD^2 statistic with Gaussian kernel

    Currently different implementations are available, in order to validate accuracy and compare speeds. The widely used median heuristic for bandwidth-selection of the Gaussian kernel is used.
    """
    if torch.isnan(X).any() or torch.isnan(Y).any():
        return torch.tensor(float("nan"))

    tic = time.time()  # noqa

    if z_score:
        X_mean = torch.mean(X, axis=0)
        X_std = torch.std(X, axis=0)
        X = (X - X_mean) / X_std
        Y = (Y - X_mean) / X_std

    n_1 = X.shape[0]
    n_2 = Y.shape[0]

    # Bandwidth
    if bandwidth == "X":
        sigma_tensor = torch.median(torch.pdist(X))
    elif bandwidth == "XY":
        sigma_tensor = torch.median(torch.pdist(torch.cat([X, Y])))
    else:
        raise NotImplementedError

    # Compute MMD
    if implementation == "tp_sutherland":
        K = tp_ExpQuadKernel(X, Y, sigma=sigma_tensor)
        statistic = tp_mmd2_unbiased(K)

    elif implementation == "tp_djolonga":
        alpha = 1 / (2 * sigma_tensor ** 2)
        test = tp_MMDStatistic(n_1, n_2)
        statistic = test(X, Y, [alpha])

    else:
        raise NotImplementedError

    toc = time.time()  # noqa
    # log.info(f"Took {toc-tic:.3f}sec")

    return statistic




def median_distance(
    predictive_samples: torch.Tensor,
    observation: torch.Tensor,
) -> torch.Tensor:
    """Compute median distance

    Uses NumPy implementation, see [1] for discussion of differences.

    Args:
        predictive_samples: Predictive samples
        observation: Observation

    Returns:
        Median distance

    [1]: https://github.com/pytorch/pytorch/issues/1837
    """
    assert predictive_samples.ndim == 2
    assert observation.ndim == 2

    l2_distance = torch.norm((observation - predictive_samples), dim=-1)
    return torch.tensor([np.median(l2_distance.numpy()).astype(np.float32)])


def posterior_mean_error(
    samples: torch.Tensor,
    reference_posterior_samples: torch.Tensor,
) -> torch.Tensor:
    """Return absolute error between posterior means normalized by true std.
    Args:
        samples: Approximate samples
        reference_posterior_samples: Reference posterior samples
    Returns:
        absolute error in posterior mean, normalized by std, averaged over dimensions.
    """
    assert samples.ndim == 2
    assert reference_posterior_samples.ndim == 2
    abs_error_per_dim = (
        samples.mean(0) - reference_posterior_samples.mean(0)
    ) / reference_posterior_samples.std(0)

    return torch.mean(abs_error_per_dim)


def posterior_variance_ratio(
    samples: torch.Tensor,
    reference_posterior_samples: torch.Tensor,
) -> torch.Tensor:
    """Return ratio of approximate and true variance, averaged over dimensions.
    Args:
        samples: Approximate samples
        reference_posterior_samples: Reference posterior samples
    Returns:
        ratio of approximate and true posterior variance, averaged over dimensions.
    """
    assert samples.ndim == 2
    assert reference_posterior_samples.ndim == 2

    ratio_per_dim = samples.var(0) / reference_posterior_samples.var(0)

    return torch.mean(ratio_per_dim)
