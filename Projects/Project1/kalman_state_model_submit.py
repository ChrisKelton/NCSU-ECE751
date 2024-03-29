__all__ = ["kalman_state_model", "roughly_constant_velocity_motion_model"]

from pathlib import Path
from typing import List, Callable, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np

from generate_random_vector_ec import generate_avg_random_vector_series_from_covariance_mat


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
    ax[0].set_title("Kalman Filter Discrete-Time, Time-Invariant State-Model", fontsize=9)
    ax[0].set_xlabel("Time (s)")
    xtick_positions = np.linspace(0, len(x), len(xticks))
    ax[0].set_xticks(xtick_positions, xticks)
    ax[0].set_ylabel("Position (m)")
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
    Q_k: Union[np.ndarray, Callable],
    F_k: Union[np.ndarray, Callable],
    G_k: Union[np.ndarray, Callable],
    C: np.ndarray,
    R_k: Union[np.ndarray, Callable],
    m_0: List[float],
    PI_0: np.ndarray,
    n_vars_to_solve: int,
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

    n_obs_to_retain = len(np.where(C == 1)[0])
    r_k = np.zeros((iterations, n_obs_to_retain))
    r_k[0] = m_0[:n_obs_to_retain]
    for k in range(1, iterations):
        u = generate_avg_random_vector_series_from_covariance_mat(
            Q_k(k - 1), samples=n_vars_to_solve, iterations=rvg_iterations, rng=rng
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

        temp_r_k = C @ x_k[k]
        r_k[k] = np.squeeze(np.add(np.column_stack(temp_r_k[:n_obs_to_retain]).T, w.T[:n_obs_to_retain]).T)
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

    F_k = lambda k: np.array(([[1, period], [0, 1]]))
    if retain_all_obs:
        C = np.array([[1, 0], [0, 1]])
    else:
        C = np.array([[1, 0], [0, 0]])
    m_0 = [p0, s0]
    PI_0 = np.eye(2)

    G_k = lambda k: np.eye(2)
    if Q_k is None:
        Q_k = lambda k: acceleration_variance * np.array(
            [[(period ** 4) / 4, (period ** 3) / 2], [(period ** 3) / 2, period ** 2]]
        )
    elif not callable(Q_k):
        Q_k = lambda k: Q_k

    x_k, r_k = kalman_state_model(
        Q_k=Q_k,
        F_k=F_k,
        G_k=G_k,
        C=C,
        R_k=R_k,
        m_0=m_0,
        PI_0=PI_0,
        n_vars_to_solve=1,
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
        plot_state_model_vs_observation_roughly_constant_velocity_model(
            x_k, r_k, p_k, fig_output_path, velocity_const=s0, xticks=xticks
        )

    return r_k, Q_k, F_k, G_k, C, R_k, m_0, PI_0, p_k, s_k


def main():
    seed = 1408
    random_vector_generator_iterations = 1
    period = 0.1
    total_time = 10  # seconds
    iterations = int(total_time / period)
    fig_output_path = Path("./kalman_state_model.png")
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
    )


if __name__ == "__main__":
    main()
