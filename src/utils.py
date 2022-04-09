import copy
from typing import List, Optional, Tuple

import botorch.sampling.qmc
from ConfigSpace import ConfigurationSpace
import numpy as np

from scipy import stats

from smac.epm.gaussian_process import GaussianProcess
from smac.epm.gp_base_prior import HorseshoePrior, LognormalPrior
from smac.epm.gp_kernels import ConstantKernel, Kernel, Matern, WhiteKernel, HammingKernel


LENGTH_SCALE_BOUNDS = (np.exp(-6.754111155189306), np.exp(0.0858637988771976))


def _continuous_kernel(cont_dims: np.ndarray) -> Matern:
    return Matern(
        np.ones([len(cont_dims)]),
        [LENGTH_SCALE_BOUNDS for _ in range(len(cont_dims))],
        nu=2.5,
        operate_on=cont_dims,
    )


def _categorical_kernel(cat_dims: np.ndarray) -> HammingKernel:
    return HammingKernel(
        np.ones([len(cat_dims)]),
        [LENGTH_SCALE_BOUNDS for _ in range(len(cat_dims))],
        operate_on=cat_dims,
    )


def _noise_kernel(rng: np.random.RandomState) -> WhiteKernel:
    return WhiteKernel(
        noise_level=1e-8,
        noise_level_bounds=(np.exp(-25), np.exp(2)),
        prior=HorseshoePrior(scale=0.1, rng=rng),
    )


def _constant_kernel(rng: np.random.RandomState) -> ConstantKernel:
    return ConstantKernel(
        2.0,
        constant_value_bounds=(np.exp(-10), np.exp(2)),
        prior=LognormalPrior(mean=0.0, sigma=1, rng=rng),
    )


def _get_continuous_and_categorical_kernels(
    dim: int,
    types: List[int],
) -> Tuple[Optional[Matern], Optional[HammingKernel]]:
    cont_dims = np.where(np.array(types) == 0)[0]
    n_cont = len(cont_dims)
    cat_dims = np.where(np.array(types) != 0)[0]
    n_cat = len(cat_dims)

    cont_kernel = _continuous_kernel(cont_dims) if n_cont > 0 else None
    cat_kernel = _categorical_kernel(cat_dims) if n_cat > 0 else None

    if n_cont + n_cat != dim:
        raise ValueError(
            "The sum of dimensions of continuous and categorical parameters must be "
            f"equal to the dimension of config space (={dim}), but got {n_cont + n_cat}"
        )

    return cont_kernel, cat_kernel


def _default_kernel(
    config_space: ConfigurationSpace,
    types: List[int],
    rng: np.random.RandomState,
) -> Kernel:
    constant_kernel = _constant_kernel(rng)
    dim = len(config_space.get_hyperparameters())
    cont_kernel, cat_kernel = _get_continuous_and_categorical_kernels(dim, types)
    noise_kernel = _noise_kernel(rng)

    if cont_kernel is not None and cat_kernel is not None:
        kernel = constant_kernel * (cont_kernel * cat_kernel) + noise_kernel
    elif cont_kernel is not None:
        kernel = constant_kernel * cont_kernel + noise_kernel
    elif cat_kernel is not None:
        kernel = constant_kernel * cat_kernel + noise_kernel

    return kernel


def get_gaussian_process(
    config_space: ConfigurationSpace,
    types: List[int],
    bounds: List[Tuple[float, float]],
    kernel: Optional[Kernel],
    rng: np.random.RandomState,
    seed: Optional[int] = None,
) -> GaussianProcess:
    """ Get a SMAC kernel class. """

    if kernel is None:
        kernel = _default_kernel(config_space, types, rng)
    else:
        kernel = copy.deepcopy(kernel)

    return GaussianProcess(
        kernel=kernel,
        normalize_y=True,
        seed=seed if seed is not None else rng.randint(0, 2 ** 20),
        types=types,
        bounds=bounds,
        configspace=config_space,
    )


def _calculate_cholesky(cov: np.ndarray, initial_noise: float) -> Optional[np.ndarray]:
    """
    Calculate the cholesky decomposition of a covariance matrix.
    By adding a small diagonal element, we can usually obtain the decomposition.
    However, in case we could not get the decomposition, we return None.

    Args:
        cov (np.ndarray):
            The covariance matrix to compute the cholesky decomposition.
        initial_noise (float):
            The minimum noise to consider in the cholesky decomposition.

    Returns:
        L (Optional[np.ndarray]):
            The lower-triangle matrix of the cholesky-decomposed covariance matrix.
            If we could not get the decomposition, we return None.
    """
    N = len(cov)
    noise = initial_noise
    while noise < 1:
        try:
            L = np.linalg.cholesky(cov + np.identity(N) * noise)
            return L
        except np.linalg.LinAlgError:
            noise *= 10

    return None


def sample_predictions_from_sobol(
    gp: GaussianProcess,
    X: np.ndarray,
    n_samples: int,
    seed: Optional[int] = None
) -> np.ndarray:
    """
    Since covariance matrices are always positive definite,
    we can decompose x.T @ cov @ x into (L @ x).T @ (L @ x).
    In this case, we can sample from cov by sampling from
    an identity covariance matrix and transform it by L.

    Args:
        gp (GaussianProcess):
            A trained gaussian process regressor.
        X (np.ndarray):
            A set of inputs that we would like to predict.
        n_samples (int):
            The number of samples.
        seed (int):
            The random seed.

    Returns:
        samples_from_gp (np.ndarray):
            Samples from the trained Gaussian process
            with the shape of (n_samples, N).
    """

    y_mean, y_cov = gp.predict(X, cov_return_type='full_cov')
    N = len(X)
    L = _calculate_cholesky(y_cov, initial_noise=1e-14)

    if L is None:  # Could not compute the cholesky decomposition
        samples_from_gp = np.tile(y_mean, reps=n_samples).T
        return samples_from_gp

    rng = botorch.sampling.qmc.NormalQMCEngine(N, seed=seed)
    samples_from_gp = y_mean.flatten() + (rng.draw(n_samples).numpy() @ L)
    return samples_from_gp


def copula_transform(values: np.ndarray) -> np.ndarray:
    """
    Calculate the copula values based on the paper.

    Args:
        values (np.ndarray):
            The output values with the shape of (N, 1).

    Returns:
        copula_values (np.ndarray):
            The copula transformed output values.

    Reference:
        Title: Copula transformation from a Quantile-based Approach for Hyperparameter Transfer Learning
        Authors: David Salinas et. al
    """
    N = len(values)
    # Winsorized cut-off estimator in Section 3
    cutoff = 1.0 / (4 * N ** 0.25) / np.sqrt(np.pi * np.log(N))
    rank = stats.rankdata(values.flatten())
    quantiles = (rank - 1) / (N - 1)
    quantiles = np.clip(quantiles, a_min=cutoff, a_max=1 - cutoff)

    # Inverse Gaussian CDF in Section 3
    copula = stats.norm.ppf(quantiles)[:, np.newaxis]
    return copula
