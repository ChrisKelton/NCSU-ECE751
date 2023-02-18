from pathlib import Path
from typing import List, Optional, Union, Tuple

import matplotlib.figure
import matplotlib.pyplot as plt
import numpy as np
import scipy


def generate_vector_series_from_covariance_mat(
    cov: np.ndarray,
    samples: int = 1000,
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
    :return:
    """

    # step a)
    # get the lower triangular matrix from the cholesky decomposition of the covariance
    L = np.linalg.cholesky(cov)

    # step b)
    # generate some samples vectors series
    X = np.random.randn(samples, cov.shape[0])
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
    P_X, L_X, U_X = scipy.linalg.lu(cov_X)
    # Here we are trying to get our sample covariance to be the identity matrix
    inv_L_X = np.linalg.inv(L_X.T)
    # solving for standard_X = L^-1 * zero_mean_X, to obtain a final X that has zero mean and an identity matrix
    # covariance matrix; i.e., unit variance
    # we will use this standard_X as our X in step c)
    # additionally divide by the number of samples - 1 to get an unbaised normalization
    standard_X = (X @ inv_L_X) * np.sqrt((X.shape[0] - 1) / (X.shape[0] - 1))

    # step c)
    Y = standard_X @ L

    return Y


def element_wise_percent_diff_between_numpy_arrays(initial: np.ndarray, final: np.ndarray) -> np.ndarray:
    return abs(np.divide(abs(np.subtract(final, initial)), initial)) * 100


def compare_generate_random_vector_from_cov_against_cov(cov: np.ndarray, samples: int = 1000) -> Tuple[np.ndarray, np.ndarray]:
    Y = generate_vector_series_from_covariance_mat(cov, samples)
    cov_Y = np.cov(Y.T)
    percent_diff_cov_Y = element_wise_percent_diff_between_numpy_arrays(cov, cov_Y)

    return percent_diff_cov_Y, cov_Y


def compare_many_generated_vector_series_from_cov_against_cov(
    cov: np.ndarray,
    samples: int = 1000,
    iterations: int = 100,
) -> Tuple[List[float], np.ndarray]:
    avg_diff = np.zeros(cov.shape)
    moving_avg_diff = []
    for it in range(iterations):
        diff, cov_Y = compare_generate_random_vector_from_cov_against_cov(cov, samples)
        avg_diff += diff
        moving_avg_diff.append(np.sum(avg_diff) / ((it + 1) * cov.shape[0] * cov.shape[1]))

    avg_diff /= iterations
    print(
        f"***Average Differences***\n"
        f"Overall Average: '{moving_avg_diff[-1]:.5f}%'\n"
        f"average:\n'{avg_diff}'"
    )
    return moving_avg_diff, cov_Y


def plot_moving_average(moving_avg: List[float], title: Optional[str] = None) -> matplotlib.figure.Figure:
    if title is None:
        title = "Moving Average of Percent Difference of Random Vector Generation from Covariance"

    fig, ax = plt.subplots()
    x = np.arange(1, len(moving_avg)+1)
    ax.plot(x, moving_avg)
    ax.set_title(title, fontsize=8)
    ax.set_xlabel("Iterations")
    ax.set_ylabel("Percent Difference")
    moving_avg_avg = np.mean(moving_avg)
    ax.legend([f"mean = {moving_avg_avg:.4f}%, samples = {len(moving_avg)}"])
    fig.tight_layout()

    return fig


def generate_many_random_vectors_and_plot(
    cov: np.ndarray, fig_output_path: Union[str, Path], samples: int = 1000, iterations: int = 100,
):
    if Path(fig_output_path).suffix in [""]:
        Path(fig_output_path).mkdir(exist_ok=True, parents=True)
        fig_output_path = Path(fig_output_path) / "percent_diff_moving_avg_of_random_series_generated_from_cov.png"
    elif not Path(fig_output_path).parent.exists():
        Path(fig_output_path).parent.mkdir(exist_ok=True, parents=True)

    moving_avg, cov_Y = compare_many_generated_vector_series_from_cov_against_cov(
        cov=cov,
        samples=samples,
        iterations=iterations,
    )
    fig = plot_moving_average(moving_avg)
    fig.savefig(str(fig_output_path.absolute()))
    plt.close()
    print(f"\nOriginal covariance:\n{cov}\n\n"
          f"Generated vector's covariance:\n{cov_Y}")


def main():
    U = np.asarray([[11 / 144, -1 / 96], [-1 / 96, 73 / 960]])
    samples = 1000
    iterations = 10000
    fig_output_path = Path("./ec")
    generate_many_random_vectors_and_plot(U,fig_output_path, samples=samples, iterations=iterations)


if __name__ == '__main__':
    main()
