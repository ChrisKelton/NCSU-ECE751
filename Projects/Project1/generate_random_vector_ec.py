__all__ = ["generate_vector_series_from_covariance_mat", "generate_avg_random_vector_series_from_covariance_mat"]

import os
import time
from pathlib import Path
from typing import List, Optional, Union, Tuple

import matplotlib.figure
import matplotlib.pyplot as plt
import numpy as np
import scipy
from joblib import Parallel, delayed
from tqdm import tqdm


def generate_vector_series_from_covariance_mat(
    cov: np.ndarray,
    samples: int = 1000,
    mu: Union[float, List[float]] = 0,
    rng: Optional[np.random._generator.Generator] = None,
) -> np.ndarray:
    """
    To generate a random vector of a specific covariance matrix R
        a) Obtain the so-called Cholesky factorization (where we depend on LU Decomposition rather than the simple
            Cholesky case of LU Decomposition [where U = L.T]):

            R = AA.T

        b) Generate IID vector X

        c) Construct Y = AX

    :param cov:
    :param samples:
    :param mu:
    :return:
    """

    # step a)
    # get the lower triangular matrix from the cholesky decomposition of the covariance (switched to lu-decomposition,
    # b/c some input covariances were not positive definite)
    P, L, U = scipy.linalg.lu(cov)

    # step b)
    # generate some samples vectors series
    if rng is None:
        rng = np.random.default_rng()
    X = rng.normal(size=(np.max((2, samples, cov.shape[0])), cov.shape[0]))
    # We now want to remove the random variation away from the zero mean and identity covariance (making the sample mean
    # 0 and sample covariance the identity matrix of cov.shape)

    # zero mean the random vector series
    X -= np.mean(X)

    # get covariance of zero mean vector series in order to get the covariance of our matrix as the identity matrix
    # np.cov() expects its input data matrix to have observations in each column, and variables in each row, so we have
    # to transpose our X vector series to get its covariance matrix in the right form
    cov_X = np.cov(X.T)
    # get lower triangular with unit diagonal elements decomposition matrix of our covariance matrix
    # use scipy.linalg.lu b/c our cov_X matrix may not be positive definite
    # Note: a matrix is positive definite if it's symmetric and all its pivots are positive (if a matrix is in
    # row-echelon form, then the first nonzero entry of each row is called a pivot) or if all eigenvalues are positive.
    # Additionally, if it's a symmetric matrix all the eigenvalues are real.
    # scipy.linalg.lu uses the LU Decomposition method rather than the cholesky method
    # consequently, we will get three matrix outputs rather than one.
    # P = permutation matrix, which is necessary when attempting to obtain nonzero pivots for all rows
    # L = lower triangular matrix with unit diagonal elements
    # U = upper triangular matrix
    # to solve the equation
    # PX = LU, we can see here that when U = L.T, we get the cholesky decomposition (no need for P then)
    if len(cov_X.shape) == 0:
        P_X, L_X, U_X = scipy.linalg.lu(np.expand_dims(np.expand_dims(cov_X, 0), 0))
    else:
        P_X, L_X, U_X = scipy.linalg.lu(cov_X)
    # Here we are trying to get our sample covariance to be the identity matrix
    inv_L_X = np.linalg.inv(L_X.T)
    # solving for standard_X = L^-1 * zero_mean_X, to obtain a final X that has zero mean and an identity matrix
    # covariance matrix; i.e., unit variance
    # we will use this standard_X as our X in step c)
    # additionally multiply by the square root of the number of samples - 1 / number of samples to get an unbaised normalization
    standard_X = (X @ inv_L_X) * np.sqrt((X.shape[0] - 1) / X.shape[0])

    # step c)
    Y = standard_X @ L
    if not isinstance(mu, list):
        mu = [mu] * Y.shape[-1]
    for idx, mu_ in enumerate(mu):
        Y[:, idx] += mu_
    if samples == 1:
        Y = np.mean(Y, axis=0)

    return Y


def generate_avg_random_vector_series_from_covariance_mat(
    cov: np.ndarray,
    mu: Union[float, List[float]] = 0,
    samples: int = 1000,
    iterations: int = 100,
    rng: Optional[np.random._generator.Generator] = None,
    verbose: bool = False,
) -> np.ndarray:
    if not isinstance(mu, list):
        mu = [mu] * cov.shape[0]
    if len(mu) != cov.shape[0]:
        if len(mu) == 1:
            mu *= cov.shape[0]
        else:
            raise ValueError(
                f"Provided mean values do not match length of covariance shape. {len(mu)} != {cov.shape[0]}"
            )
    if rng is None:
        rng = np.random.default_rng()

    avg_Y = np.zeros((iterations, np.max((samples, cov.shape[0])), cov.shape[0]))
    if samples == 1 and cov.shape[0] == 1:
        if iterations >= 100000:
            t = []
            t.append(Parallel(n_jobs=os.cpu_count())(delayed(rng.normal)(mu[0], cov) for _ in range(iterations)))
            avg_Y = np.squeeze(np.array(t))
        else:
            if verbose:
                for it in tqdm(range(iterations), desc="Generating Random Vector Series"):
                    avg_Y[it] = rng.normal(mu[0], cov)
            else:
                for it in range(iterations):
                    avg_Y[it] = rng.normal(mu[0], cov)
    else:
        if iterations > 100000:
            t = []
            t.append(
                Parallel(n_jobs=4)(
                    delayed(generate_vector_series_from_covariance_mat)(cov, samples, mu, rng)
                    for _ in range(iterations)
                )
            )
            avg_Y = np.squeeze(np.array(t))
        else:
            if verbose:
                for it in tqdm(range(iterations), desc="Generating Random Vector Series"):
                    avg_Y[it] = generate_vector_series_from_covariance_mat(cov, samples, mu, rng=rng)
            else:
                for it in range(iterations):
                    avg_Y[it] = generate_vector_series_from_covariance_mat(cov, samples, mu, rng=rng)

    avg_Y = np.mean(avg_Y, axis=0)
    if samples < cov.shape[0]:
        avg_Y = np.expand_dims(np.mean(avg_Y, axis=0), 0)

    return avg_Y


def element_wise_percent_diff_between_numpy_arrays(initial: np.ndarray, final: np.ndarray) -> np.ndarray:
    return abs(np.divide(abs(np.subtract(final, initial)), initial)) * 100


def compare_generate_random_vector_from_cov_against_cov(
    cov: np.ndarray, samples: int = 1000
) -> Tuple[np.ndarray, np.ndarray]:
    Y = generate_vector_series_from_covariance_mat(cov, samples)
    cov_Y = np.cov(Y.T)
    percent_diff_cov_Y = element_wise_percent_diff_between_numpy_arrays(cov, cov_Y)

    return percent_diff_cov_Y, cov_Y


def compare_many_generated_vector_series_from_cov_against_cov(
    cov: np.ndarray, samples: int = 1000, iterations: int = 100
) -> Tuple[List[float], np.ndarray]:
    avg_diff = np.zeros(cov.shape)
    cov_avg = np.zeros(cov.shape)
    moving_avg_diff = []
    for it in tqdm(range(iterations), desc="Generating random variable(s) from covariance"):
        diff, cov_Y = compare_generate_random_vector_from_cov_against_cov(cov, samples)
        avg_diff += diff
        cov_avg = np.add(cov_avg, cov_Y)
        moving_avg_diff.append(np.sum(avg_diff) / ((it + 1) * cov.shape[0] * cov.shape[1]))

    avg_diff /= iterations
    print(f"***Average Differences***\n" f"Overall Average: '{moving_avg_diff[-1]:.5f}%'\n" f"average:\n'{avg_diff}'")
    return moving_avg_diff, np.divide(cov_avg, iterations)


def plot_moving_average(moving_avg: List[float], title: Optional[str] = None) -> matplotlib.figure.Figure:
    if title is None:
        title = "Moving Average of Percent Difference of Random Vector Generation from Covariance"

    fig, ax = plt.subplots()
    x = np.arange(1, len(moving_avg) + 1)
    ax.plot(x, moving_avg)
    ax.set_title(title, fontsize=8)
    ax.set_xlabel("Iterations")
    ax.set_ylabel("Percent Difference")
    moving_avg_avg = np.nanmean(np.ma.masked_invalid(moving_avg))
    ax.legend([f"mean = {moving_avg_avg:.4f}%, samples = {len(moving_avg)}"])
    fig.tight_layout()

    return fig


def generate_many_random_vectors_and_plot(
    cov: np.ndarray, fig_output_path: Union[str, Path], samples: int = 1000, iterations: int = 100
):
    if Path(fig_output_path).suffix in [""]:
        Path(fig_output_path).mkdir(exist_ok=True, parents=True)
        fig_output_path = Path(fig_output_path) / "percent_diff_moving_avg_of_random_series_generated_from_cov.png"
    elif not Path(fig_output_path).parent.exists():
        Path(fig_output_path).parent.mkdir(exist_ok=True, parents=True)

    moving_avg, cov_avg_Y = compare_many_generated_vector_series_from_cov_against_cov(
        cov=cov, samples=samples, iterations=iterations
    )
    fig = plot_moving_average(moving_avg)
    fig.savefig(str(fig_output_path.absolute()))
    plt.close()
    print(f"\nOriginal covariance:\n{cov}\n\n" f"Generated vector's covariance:\n{cov_avg_Y}")


def main(fig_output_path: Path, samples: int = 1000, number_of_random_variables: int = 2, iterations: int = 10000):
    x = np.random.normal(size=[samples, number_of_random_variables])
    U = np.cov(x.T)
    generate_many_random_vectors_and_plot(U, fig_output_path, samples=samples, iterations=iterations)


def test_parallelizing(samples: int = 1000, number_of_random_variables: int = 2, iterations: int = 10000):
    x = np.random.normal(size=[samples, number_of_random_variables])
    U = np.cov(x.T)
    rng = np.random.default_rng()
    print(f"iterations = '{iterations}'")
    start_time_unparallel = time.time_ns() / (10 ** 9)  # convert to floating-point seconds
    temp = np.zeros((iterations, np.max((samples, U.shape[0])), U.shape[0]))
    for it in range(iterations):
        temp[it] = generate_vector_series_from_covariance_mat(U, samples, 0, rng)
    end_time_unparallel = time.time_ns() / (10 ** 9)
    tt_unparallel = end_time_unparallel - start_time_unparallel
    print(f"total time unparallel: '{tt_unparallel} s'  ---  '{tt_unparallel / 60} min'")

    start_time_parallel = time.time_ns() / (10 ** 9)
    t = []
    t.append(
        Parallel(n_jobs=os.cpu_count())(
            delayed(generate_vector_series_from_covariance_mat)(U, samples, 0, rng) for _ in range(iterations)
        )
    )
    Y = np.squeeze(np.array(t))
    end_time_parallel = time.time_ns() / (10 ** 9)
    tt_parallel = end_time_parallel - start_time_parallel
    print(f"total time parallel: '{tt_parallel} s'  ---  '{tt_parallel / 60} min'")


if __name__ == "__main__":
    samples = 10000
    number_of_random_variables = 4
    iterations = 100000
    # test_parallelizing(iterations=iterations)
    fig_output_path = Path("./ec")
    main(
        fig_output_path=fig_output_path,
        samples=samples,
        number_of_random_variables=number_of_random_variables,
        iterations=iterations,
    )
