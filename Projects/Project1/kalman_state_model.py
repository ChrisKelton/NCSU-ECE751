from enum import Enum
from pathlib import Path
from typing import List, Callable, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np

from generate_random_vector_ec import generate_avg_random_vector_series_from_covariance_mat


class RoughlyConstantStateModel(Enum):
    ConstantVelocityModelPosition0 = "0"
    ConstantVelocityModelPosition1 = "1"
    ConstantAccelerationModelPosition = "2"
    ConstantAccelerationModelVelocityAndPosition = "3"


def plot_state_model_vs_observation_roughly_constant_velocity_model(
    x_k: np.ndarray,
    r_k: np.ndarray,
    position_ground_truth: List[float],
    fig_output_path: Path,
    velocity_const: Optional[float] = None,
    xticks: Optional[Union[list, np.ndarray]] = None,
):
    x = np.arange(0, len(x_k))
    fig, ax = plt.subplots(2, 1, figsize=(10, 10))
    ax[0].plot(x, position_ground_truth, label="truth")
    ax[0].plot(x, r_k[:, 0], "-.", label="state model position")
    # ax[0].plot(x, r_k, label="observation position")
    ax[0].set_title("Kalman Filter Discrete-Time, Time-Invariant State-Model", fontsize=9)
    ax[0].set_xlabel("Time (s)")
    xtick_positions = np.linspace(0, len(x), len(xticks))
    ax[0].set_xticks(xtick_positions, xticks)
    ax[0].set_ylabel("Position (m)")
    # ax[0].legend(["state model position", "truth", "observation position"])
    ax[0].legend(["truth", "state model position"])

    legend = []
    if velocity_const is not None:
        ax[1].plot(x, [velocity_const] * len(x), label="velocity constant")
        legend.append("velocity constant")
    legend.append("state estimate constant velocity")
    ax[1].plot(x, x_k[:, 1], "-.", label="state estimate constant velocity")
    ax[1].set_xlabel("Time (s)")
    ax[1].set_xticks(xtick_positions, xticks)
    ax[1].set_ylabel("Velocity (m/s)")
    ax[1].legend(legend)

    fig.tight_layout()
    fig.savefig(str(fig_output_path.absolute()))
    plt.close()


def plot_state_model_vs_observation_roughly_constant_acceleration_model(
    x_k: np.ndarray,
    r_k: np.ndarray,
    position_ground_truth: List[float],
    velocity_ground_truth: List[float],
    fig_output_path: Path,
    acceleration_const: Optional[float] = None,
    xticks: Optional[Union[list, np.ndarray]] = None,
):
    x = np.arange(0, len(x_k))
    fig, ax = plt.subplots(3, 1, figsize=(15, 15))
    ax[0].plot(x, r_k[:, 0], "-.", label="state model position")
    ax[0].plot(x, position_ground_truth, label="truth")
    # ax[0].plot(x, r_k[:, 0], label="observation position")
    ax[0].set_xlabel("Time (s)")
    xtick_positions = np.linspace(0, len(x), len(xticks))
    ax[0].set_xticks(xtick_positions, xticks)
    ax[0].set_ylabel("Position (m)")
    ax[0].legend(["state model position", "truth"])

    legend = []
    if r_k.shape[1] > 1:
        ax[1].plot(x, r_k[:, 1], "-.", label="observation velocity")
        legend.append("observation velocity")
    else:
        ax[1].plot(x, x_k[:, 1], "-.", label="state model velocity")
        legend.append("state model velocity")
    ax[1].plot(x, velocity_ground_truth, label="truth")
    legend.append("truth")
    ax[1].legend(legend)
    ax[1].set_title("Roughly Constant Acceleration")
    ax[1].set_xlabel("Time (s)")
    ax[1].set_xticks(xtick_positions, xticks)
    ax[1].set_ylabel("Velocity (m/s)")

    legend = []
    if acceleration_const is not None:
        ax[2].plot(x, [acceleration_const] * len(x), label="acceleration constant")
        legend.append("acceleration constant")
    legend.append("state model constant acceleration")
    ax[2].plot(x, x_k[:, 2], "-.", label="state model acceleration")
    ax[2].set_xlabel("Time (s)")
    ax[2].set_xticks(xtick_positions, xticks)
    ax[2].set_ylabel("Acceleration (m/s^2)")
    ax[2].legend(legend)

    fig.tight_layout()
    fig.savefig(str(fig_output_path.absolute()))
    plt.close()


def initial_state_vector(
    m_0: List[float] = 0.0,
    PI_0: np.ndarray = np.zeros((1,)),
    samples: int = 2,
    iterations: int = 1000,
    rng: Optional[np.random._generator.Generator] = None,
) -> np.ndarray:
    return generate_avg_random_vector_series_from_covariance_mat(
        PI_0, m_0, samples=samples, iterations=iterations, rng=rng
    )


def kalman_state_model(
    Q_k: Callable,
    F_k: Callable,
    G_k: Callable,
    C_k: np.ndarray,
    R_k: Callable,
    m_0: List[float],
    PI_0: np.ndarray,
    iterations: int = 100,
    rvg_iterations: int = 1000,
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    if seed is not None:
        rng = np.random.default_rng(seed)
    else:
        rng = np.random.default_rng()

    if not callable(Q_k):
        Q_k = lambda k: Q_k
    if not callable(F_k):
        F_k = lambda k: F_k
    if not callable(G_k):
        G_k = lambda k: G_k
    if not callable(R_k):
        R_k = lambda k: R_k

    if m_0 is None:
        m_0 = [0]
    x_k = np.zeros((iterations, len(m_0)))
    if PI_0.any():
        x_0 = initial_state_vector(m_0, PI_0, samples=1, iterations=rvg_iterations, rng=rng)
    else:
        x_0 = m_0
    x_k[0] = x_0

    # u_k = generate_avg_random_vector_series_from_covariance_mat(Q_k, samples=iterations, iterations=rvg_iterations, seed=seed)
    # w_k = generate_avg_random_vector_series_from_covariance_mat(R_k, samples=iterations, iterations=rvg_iterations, seed=seed)
    vars_to_solve = len(np.where(C_k == 1)[0])
    # u_k = np.zeros((iterations, vars_to_solve))  # determined by how many variables we are solving for
    # w_k = np.zeros((iterations, vars_to_solve))
    r_k = np.zeros((iterations, vars_to_solve))
    r_k[0] = m_0[:vars_to_solve]
    for k in range(1, iterations):
        u = generate_avg_random_vector_series_from_covariance_mat(
            Q_k(k - 1), samples=vars_to_solve, iterations=rvg_iterations, rng=rng
        )
        w = generate_avg_random_vector_series_from_covariance_mat(
            R_k(k - 1), samples=1, iterations=rvg_iterations, rng=rng
        )
        # print(f"k = '{k}'")
        s_k = np.expand_dims(F_k(k) @ x_k[k - 1], 1)

        if G_k(k).shape[1] == u.shape[0]:
            u_k_1 = G_k(k) @ u
            x_k[k] = np.squeeze(np.add(s_k, u_k_1))
        elif G_k(k).shape[0] == u.shape[1]:
            u_k_1 = u @ G_k(k)
            x_k[k] = np.squeeze(np.add(s_k, u_k_1.T))
        else:
            raise RuntimeError(
                f"Mismatched shapes between u_k & G_k(k) where k = '{k}'.\n"
                f"G_k(k) = \n"
                f"'{G_k(k)}'\n"
                f"u = \n"
                f"'{u}'"
            )

        # print(f"x_k[k] = '{x_k[k]}'")

        temp_r_k = C_k @ x_k[k]
        r_k[k] = np.squeeze(np.add(np.column_stack(temp_r_k[:vars_to_solve]).T, w.T[:vars_to_solve]).T)
        # print(f"r_k[k] = '{r_k[k]}'\n\n")

    return x_k, r_k


def roughly_constant_velocity_motion_model(
    p0: float,
    s0: float,
    a0: float = 0,
    period: float = 1,
    acceleration_variance: Optional[float] = None,
    variance_R_k: Optional[float] = None,
    Q_k: Optional[Union[np.ndarray, Callable]] = None,
    R_k: Optional[Union[np.ndarray, Callable]] = None,
    iterations: int = 100,
    rvg_iterations: int = 1000,
    seed: Optional[int] = None,
    fig_output_path: Optional[Path] = None,
    method: Union[int, RoughlyConstantStateModel] = RoughlyConstantStateModel.ConstantVelocityModelPosition0,
    retain_all_obs: bool = False,
) -> Tuple[
    np.ndarray,
    Callable,
    Callable,
    Callable,
    np.ndarray,
    Callable,
    List[float],
    np.ndarray,
    List[float],
    List[float],
]:
    if isinstance(method, int):
        method = RoughlyConstantStateModel(str(method))
    if seed is not None:
        rng = np.random.default_rng(seed)
    else:
        rng = np.random.default_rng()
    if acceleration_variance is None:
        acceleration_variance = rng.uniform(0, 1, size=1)
    if variance_R_k is None:
        variance_R_k = rng.uniform(0, 1, size=1)

    if R_k is None:
        R_k = lambda k: np.eye(2) * np.asarray((variance_R_k, 1))
    elif not callable(R_k):
        R_k = lambda k: R_k

    if (
        method is RoughlyConstantStateModel.ConstantVelocityModelPosition0
        or method == RoughlyConstantStateModel.ConstantVelocityModelPosition1
    ):
        F_k = lambda k: np.array(([[1, period], [0, 1]]))
        if retain_all_obs:
            C = np.array([[1, 0], [0, 1]])
        else:
            C = np.array([[1, 0], [0, 0]])
        m_0 = [p0, s0]
        # PI_0 = np.zeros((2, 2))
        PI_0 = np.eye(2)
    elif (
        method == RoughlyConstantStateModel.ConstantAccelerationModelPosition
        or method == RoughlyConstantStateModel.ConstantAccelerationModelVelocityAndPosition
    ):
        F_k = lambda k: np.array(([[1, period, (period ** 2) / 2], [0, 1, period], [0, 0, 1]]))
        if retain_all_obs:
            C = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        elif method == 2:
            C = np.array([[1, 0, 0], [0, 0, 0], [0, 0, 0]])
        else:
            C = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 0]])
        m_0 = [p0, s0, a0]
        # PI_0 = np.zeros((3, 3))
        PI_0 = np.eye(3)
    else:
        raise RuntimeError(f"Got unsupported method '{method}'. Choose from: '0', '1', '2', or '3'.")

    if method == RoughlyConstantStateModel.ConstantVelocityModelPosition0:
        G_k = lambda k: np.array(([[0.5 * (period ** 2)], [period]]))
        if Q_k is None:
            Q_k = lambda k: np.eye(1) * acceleration_variance
    elif method == RoughlyConstantStateModel.ConstantVelocityModelPosition1:
        G_k = lambda k: np.eye(2)
        if Q_k is None:
            Q_k = lambda k: acceleration_variance * np.array(
                [[(period ** 4) / 4, (period ** 3) / 2], [(period ** 3) / 2, period ** 2]]
            )
    elif method == RoughlyConstantStateModel.ConstantAccelerationModelPosition:
        raise NotImplementedError(f"{method.name} Not Implemented")
        G_k = lambda k: np.array(([[(period ** 2) / 2], [period], [1]]))
        if Q_k is None:
            Q_k = lambda k: np.eye(1) * acceleration_variance
    elif method == RoughlyConstantStateModel.ConstantAccelerationModelVelocityAndPosition:
        G_k = lambda k: np.eye(3)
        if Q_k is None:
            Q_k = lambda k: np.array(
                (
                    [
                        [(period ** 4) / 4, (period ** 3) / 2, (period ** 2) / 2],
                        [(period ** 3) / 2, period ** 2, period],
                        [(period ** 2) / 2, period, 1],
                    ]
                )
            )
    else:
        raise RuntimeError(f"Got unsupported method '{method}'. Choose from: '0', '1', '2', or '3'.")

    if not callable(Q_k):
        Q_k = lambda k: Q_k

    x_k, r_k = kalman_state_model(
        Q_k=Q_k,
        F_k=F_k,
        G_k=G_k,
        C_k=C,
        R_k=R_k,
        m_0=m_0,
        PI_0=PI_0,
        iterations=iterations,
        rvg_iterations=rvg_iterations,
        seed=seed,
    )
    # ground truth
    # p_k = p_k-1 + period*v_k-1 + 0.5*a*period^2; a = 0
    # p_k = p_k-1 + period*v_k-1
    p_k = [p0]
    for _ in range(1, iterations):
        p_k.append(p_k[-1] + period * s0)
    p_k_not_constant_velocity = [p0]
    s_k = [s0]
    for _ in range(1, iterations):
        p_k_not_constant_velocity.append(p_k_not_constant_velocity[-1] + period * s_k[-1] + (0.5 * a0 * (period ** 2)))
        s_k.append(s_k[-1] + a0 * period)
    total_time = iterations * period
    xticks = np.arange(0, total_time + 1)
    if fig_output_path is not None:
        if Path(fig_output_path).is_dir():
            Path(fig_output_path).mkdir(exist_ok=True, parents=True)
            fig_output_path = Path(fig_output_path) / "kalman_filter.png"
        if (
            method == RoughlyConstantStateModel.ConstantVelocityModelPosition0
            or method == RoughlyConstantStateModel.ConstantVelocityModelPosition1
        ):
            plot_state_model_vs_observation_roughly_constant_velocity_model(
                x_k, r_k, p_k, fig_output_path, velocity_const=s0, xticks=xticks
            )
        else:
            plot_state_model_vs_observation_roughly_constant_acceleration_model(
                x_k, r_k, p_k_not_constant_velocity, s_k, fig_output_path, acceleration_const=a0, xticks=xticks
            )

    return r_k, Q_k, F_k, G_k, C, R_k, m_0, PI_0, p_k, s_k

def main():
    seed = 1408
    random_vector_generator_iterations = 1
    period = 0.1
    total_time = 10  # seconds
    iterations = int(total_time / period)
    # seed = None
    fig_output_path = Path("./kalman_state_model.png")
    # RoughlyConstantStateModel.ConstantVelocityModelPosition1 or 1 is the preferred method
    method = (
        RoughlyConstantStateModel.ConstantVelocityModelPosition1
    )  # can also be an integer: '0', '1', or '3' ('2' not implemented)
    roughly_constant_velocity_motion_model(
        p0=1000,
        s0=-50,
        a0=0,
        period=period,
        acceleration_variance=40,
        variance_R_k=100,
        iterations=iterations,
        rvg_iterations=random_vector_generator_iterations,
        seed=seed,
        fig_output_path=fig_output_path,
        method=method,
    )
    # roughly_constant_velocity_motion_model(
    #     p0=2,
    #     s0=3,
    #     a0=0,
    #     period=1,
    #     Q_k=None,
    #     R_k=None,
    #     rvg_iterations=random_vector_generator_iterations,
    #     seed=seed,
    #     fig_output_path=fig_output_path,
    #     method=0,
    # )


if __name__ == "__main__":
    main()
