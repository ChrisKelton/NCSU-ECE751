from pathlib import Path
from typing import Optional, Callable, Tuple, List, Union

import matplotlib.pyplot as plt
import numpy as np
from time import time_ns

from kalman_state_model_submit import roughly_constant_velocity_motion_model


def plot_kalman_filter_estimates_vs_observations_and_ground_truth(
    observations: np.ndarray,
    estimates: np.ndarray,
    ground_truth: List[float],
    fig_output_path: Path,
    xticks: Optional[Union[list, np.ndarray]] = None,
):
    x = np.arange(0, len(observations))
    fig, ax = plt.subplots(2, 1, figsize=(10, 10))
    ax[0].plot(x, ground_truth[:, 0], label="truth")
    ax[0].plot(x, observations[:, 0], "-.", label="observations")
    ax[0].plot(x, estimates[:, 0], "--", label="estimates")
    ax[0].set_title("Kalman Filter Discrete-Time, Time-Invariant", fontsize=9)
    ax[0].set_xlabel("Time (s)")
    if xticks is None:
        xticks = x
    xtick_positions = np.linspace(0, len(x), len(xticks))
    ax[0].set_xticks(xtick_positions, xticks)
    ax[0].set_ylabel("Position (m)")
    ax[0].legend(["truth", "observations", "estimates"])

    ax[1].plot(x, ground_truth[:, 1], label="truth")
    ax[1].plot(x, observations[:, 1], "-.", label="observations")
    ax[1].plot(x, estimates[:, 1], "--", label="estimates")
    ax[1].set_title("Kalman Filter Discrete-Time, Time-Invariant", fontsize=9)
    ax[1].set_xlabel("Time (s)")
    ax[1].set_xticks(xtick_positions, xticks)
    ax[1].set_ylabel("Velocity (m/s)")
    ax[1].legend(["truth", "observations", "estimates"])

    fig.tight_layout()
    fig.savefig(fig_output_path)
    plt.close()


def plot_kalman_filter_kalman_gain_position_and_velocity(
    K: np.ndarray,
    fig_output_path: Path,
    xticks: Optional[Union[list, np.ndarray]] = None,
):
    x = np.arange(0, len(K))
    fig, ax = plt.subplots(2, 1, figsize=(10, 10))
    ax[0].plot(x, K[:, 0])
    ax[0].set_title("Kalman Gain Position")
    ax[0].set_xlabel("Time (s)")
    if xticks is None:
        xticks = x
    xtick_positions = np.linspace(0, len(x), len(xticks))
    ax[0].set_xticks(xtick_positions, xticks)
    ax[0].set_ylabel("Gain")
    ax[0].set_ylim([0, 1.5])

    ax[1].plot(x, K[:, 1])
    ax[1].set_title("Kalman Gain Velocity")
    ax[1].set_xlabel("Time (s)")
    ax[1].set_xticks(xtick_positions, xticks)
    ax[1].set_ylabel("Gain")
    ax[1].set_ylim([0, 1.5])

    fig.tight_layout()
    fig.savefig(fig_output_path)
    plt.close()


def plot_kalman_filter_estimate_covariance_diagonals(
    P: np.ndarray,
    fig_output_path: Path,
    xticks: Optional[Union[list, np.ndarray]] = None,
):
    x = np.arange(0, len(P))
    fig, ax = plt.subplots(2, 1, figsize=(10, 10))
    ax[0].plot(x, P[:, 0])
    ax[0].set_title("Estimate Covariance Position Diagonals")
    ax[0].set_xlabel("Time (s)")
    if xticks is None:
        xticks = x
    xtick_positions = np.linspace(0, len(x), len(xticks))
    ax[0].set_xticks(xtick_positions, xticks)
    ax[0].set_ylabel("MSE (Position)")
    ax[0].set_yscale("log")

    ax[1].plot(x, P[:, 1])
    ax[1].set_title("Estimate Covariance Velocity Diagonals")
    ax[1].set_xlabel("Time (s)")
    ax[1].set_xticks(xtick_positions, xticks)
    ax[1].set_ylabel("MSE (Velocity)")
    ax[1].set_yscale("log")

    fig.tight_layout()
    fig.savefig(fig_output_path)
    plt.close()


def kalman_filter(
    initial_state_estimate: np.ndarray,
    P_initial: np.ndarray,
    observations: np.ndarray,
    Q_k: Callable,
    F_k: Callable,
    G_k: Callable,
    C: np.ndarray,
    R_k: Callable,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    # initial state estimate
    x_estimator = np.zeros((observations.shape[0], len(initial_state_estimate), 1))
    x_estimator[0] = initial_state_estimate

    # covariance matrix of estimates
    P = np.zeros((observations.shape[0], len(P_initial), len(P_initial)))
    P[0] = np.diag(P_initial)

    # Kalman gain
    # K = np.zeros((observations.shape[0] - 1, 1, len(initial_state_estimate)))
    K = np.zeros((observations.shape[0] - 1, len(initial_state_estimate), len(initial_state_estimate)))

    for k in range(1, len(observations)):
        # x^(k-1) = x_estimator[k-1]
        # x^(k|k-1) = F(k) x^(k-1)
        # x_est_lag = x^(k|k-1)
        x_est_lag = F_k(k) @ x_estimator[k - 1]

        # P(k-1|k) = P[k-1]
        # P(k|k-1) = F(k) P(k-1|k) F(k).T + G(k) Q(k) G(k).T
        # P_lag = P(k|k-1)
        # P_lag = np.add(F_k(k) @ P[k - 1] @ F_k(k).T, G_k(k) @ Q_k(k) @ G_k(k).T)
        P_lag = np.add((F_k(k).dot(P[k-1])).dot(F_k(k).T), G_k(k) @ Q_k(k) @ G_k(k).T)

        # K(k) = P(k|k-1) C.T [C P(k|k-1) C.T + R(k)]^-1
        # update Kalman gain, k-1 = k for K [kalman gain]
        # K[k-1] = np.diag(P_lag @ C.T @ np.linalg.inv(np.add(C @ P_lag @ C.T, R_k(k))))
        # K[k-1] = np.multiply(P_lag.dot(C.T), np.linalg.inv(np.add((C.dot(P_lag)).dot(C.T), R_k(k))))
        K[k-1] = np.multiply(P_lag.dot(C.T), np.linalg.inv(np.add((C.dot(P_lag)).dot(C.T), R_k(k))))

        # P(k) = P(k|k-1) - K(k) C P(k|k-1)
        # P(k) = [I - K(k) C] P(k|k-1); I = identity matrix
        # update estimate covariance
        P[k] = np.subtract(np.identity(C.shape[1]), K[k-1].dot(C)).dot(P_lag)

        # x^(k) = x^(k|k-1) + K(k) [r(k) - C x^(k|k-1)]
        # update state estimate
        # x_estimator[k] = np.add(x_est_lag, (K[k-1].dot(np.subtract(np.column_stack(observations[k]), C.dot(x_est_lag)))).T)
        x_estimator[k] = np.add(x_est_lag, (K[k-1].dot(np.subtract(np.row_stack(observations[k]), C.dot(x_est_lag)))))
        # x_estimator[k] = np.add(x_est_lag, np.expand_dims(K[k-1].dot(np.subtract(np.column_stack(observations[k]), C.dot(x_est_lag)))[-1, :], 0).T)

    # get estimates in same form as observations, e.g. obs[0] = [0, 0], est[0] = [[0], [0]] -> est[0, 0]
    return np.column_stack(x_estimator).T, P, K


def roughly_constant_velocity_motion_filter(
    initial_state_estimate: np.ndarray,
    P_initial: np.ndarray,  # values of P_initial should be values for diagonals of initial P matrix
    p0: float,
    s0: float,
    a0: float = 0,
    period: float = 1,
    acceleration_variance: Optional[float] = None,
    variance_R_k: Optional[float] = None,
    Q_k: Optional[Union[np.ndarray, Callable]] = None,
    R_k: Optional[Union[np.ndarray, Callable]] = None,
    n_observations: int = 100,
    rvg_iterations: int = 100,
    seed: Optional[int] = None,
    fig_output_path: Optional[Path] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    # p_k = ground truth position, s_k = ground truth position
    t0 = time_ns()
    print("*** Generating Kalman State Model Observations ***")
    observations, Q_k, F_k, G_k, C, R_k, m_0, PI_0, p_k, s_k = roughly_constant_velocity_motion_model(
        p0=p0,
        s0=s0,
        a0=a0,
        period=period,
        acceleration_variance=acceleration_variance,
        variance_R_k=variance_R_k,
        Q_k=Q_k,
        R_k=R_k,
        iterations=n_observations,
        rvg_iterations=rvg_iterations,
        seed=seed,
        fig_output_path=fig_output_path,
        retain_all_obs=True,
    )
    t1 = time_ns()
    print(f"Time taken -- {(t1 - t0)/1e9:.3f}s\n")

    t0 = time_ns()
    print("*** Estimating signal using observations ***")
    x_estimator, P, K = kalman_filter(
        initial_state_estimate=initial_state_estimate,
        P_initial=P_initial,
        observations=observations,
        Q_k=Q_k,
        F_k=F_k,
        G_k=G_k,
        C=C,
        R_k=R_k,
    )
    t1 = time_ns()
    print(f"Time taken -- {(t1 - t0)/1e9:.3f}s\n")

    if fig_output_path is not None:
        print("*** Plotting ***")
        total_time = n_observations * period
        xticks = np.arange(0, total_time + 1)
        if Path(fig_output_path).is_dir():
            Path(fig_output_path).mkdir(exist_ok=True, parents=True)
            temp_path = Path(fig_output_path) / "kalman_filter_position.png"
        else:
            temp_path = fig_output_path
            fig_output_path.parent.mkdir(exist_ok=True, parents=True)
        plot_kalman_filter_estimates_vs_observations_and_ground_truth(
            observations=observations,
            estimates=x_estimator,
            ground_truth=np.column_stack((p_k, s_k)),
            fig_output_path=temp_path,
            xticks=xticks,
        )
        temp_path = temp_path.parent / "kalman_filter_gain.png"
        plot_kalman_filter_kalman_gain_position_and_velocity(
            K=np.diagonal(K, axis1=1, axis2=2),  # get diagonals of Kalman Gain in the form of [position_diag, velocity_diag]
            fig_output_path=temp_path,
            xticks=xticks[1:],
        )
        temp_path = temp_path.parent / "kalman_filter_estimate_covariance_diagonals.png"
        plot_kalman_filter_estimate_covariance_diagonals(
            P=np.diagonal(P, axis1=1, axis2=2),
            fig_output_path=temp_path,
            xticks=xticks,
        )

    return observations, x_estimator, P, K


def main():
    seed = 1408
    random_vector_generator_iterations = 1
    period = 0.1
    total_time = 10  # seconds
    n_observations = int(total_time / period)
    fig_output_path = Path("./kalman_state_filter.png")
    initial_state_estimate = np.expand_dims(np.array((0, 0)), 1)  # 2 x 1
    P_initial = np.array((1000, 1000))
    obs, est, P, K = roughly_constant_velocity_motion_filter(
        initial_state_estimate=initial_state_estimate,
        P_initial=P_initial,
        p0=1000,
        s0=-50,
        a0=0,
        period=period,
        acceleration_variance=40,
        variance_R_k=100,
        n_observations=n_observations,
        rvg_iterations=random_vector_generator_iterations,
        seed=seed,
        fig_output_path=fig_output_path,
    )


if __name__ == '__main__':
    main()
