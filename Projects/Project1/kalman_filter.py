from pathlib import Path
from typing import List, Callable, Optional, Tuple
import matplotlib.pyplot as plt
import numpy as np

from generate_random_vector_ec import generate_avg_random_vector_series_from_covariance_mat


def plot_state_model_vs_observation_roughly_constant_velocity_model(
    x_k: np.ndarray, r_k: np.ndarray, fig_output_path: Path
):
    x = np.arange(0, len(x_k))
    plt.subplot(1, 2, 1)
    plt.plot(x, x_k[:, 0], label="state model position")
    plt.plot(x, r_k, label="observation position")
    plt.title("Kalman Filter Discrete-Time, Time-Invariant State-Model vs Observation Measurements", fontsize=9)
    plt.xlabel("k iterations")
    plt.ylabel("Position/Velocity")
    plt.legend(["state model position", "observation position"])

    plt.subplot(1, 2, 2)
    plt.plot(x, x_k[:, 1], label="state model velocity")
    plt.xlabel("k iterations")
    plt.ylabel("Velocity")

    plt.tight_layout()
    plt.savefig(str(fig_output_path.absolute()))


def initial_state_vector(
    m_0: List[float] = 0.0,
    PI_0: np.ndarray = np.zeros((1,)),
    iterations: int = 1000,
    rng: Optional[np.random._generator.Generator] = None,
) -> np.ndarray:
    return generate_avg_random_vector_series_from_covariance_mat(PI_0, m_0, samples=2, iterations=iterations, rng=rng)


def kalman_filter(
    Q_k: Callable,
    F_k: Callable,
    G_k: Callable,
    C_k: np.ndarray,
    R_k: np.ndarray,
    iterations: int = 100,
    rvg_iterations: int = 1000,
    m_0: List[float] = None,
    PI_0: np.ndarray = np.zeros((2, 2)),
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    if seed is not None:
        rng = np.random.default_rng(seed)
    else:
        rng = np.random.default_rng()

    if m_0 is None:
        m_0 = [0]
    x_k = np.zeros((iterations, len(m_0)))
    if PI_0.any():
        x_0 = initial_state_vector(m_0, PI_0, iterations=rvg_iterations, rng=rng)
    else:
        x_0 = m_0
    x_k[0] = x_0

    # u_k = generate_avg_random_vector_series_from_covariance_mat(Q_k, samples=iterations, iterations=rvg_iterations, seed=seed)
    # w_k = generate_avg_random_vector_series_from_covariance_mat(R_k, samples=iterations, iterations=rvg_iterations, seed=seed)
    vars_to_solve = len(np.where(C_k == 1)[0])
    # u_k = np.zeros((iterations, vars_to_solve))  # determined by how many variables we are solving for
    # w_k = np.zeros((iterations, vars_to_solve))
    r_k = np.zeros((iterations, len(m_0) - 1))
    for k in range(1, iterations):
        u = generate_avg_random_vector_series_from_covariance_mat(
            Q_k(k - 1), samples=vars_to_solve, iterations=rvg_iterations, rng=rng
        )
        w = generate_avg_random_vector_series_from_covariance_mat(
            R_k, samples=vars_to_solve, iterations=rvg_iterations, rng=rng
        )
        print(f"k = '{k}'")
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

        print(f"x_k[k] = '{x_k[k]}'")

        temp_r_k = C_k @ x_k[k]
        r_k[k] = np.add(temp_r_k, w)
        print(f"r_k[k] = '{r_k[k]}'\n\n")

    return x_k, r_k


def roughly_constant_velocity_motion_model(
    p0: float,
    s0: float,
    period: int,
    acceleration_variance: Optional[float] = None,
    Q_k: Optional[np.ndarray] = None,
    R_k: Optional[np.ndarray] = None,
    iterations: int = 100,
    rvg_iterations: int = 1000,
    seed: Optional[int] = None,
    fig_output_path: Optional[Path] = None,
    method: int = 0,
):
    if seed is not None:
        rng = np.random.default_rng(seed)
    else:
        rng = np.random.default_rng()
    if acceleration_variance is None:
        acceleration_variance = rng.uniform(0, 1, size=1)

    if method == 0:
        G_k = lambda k: np.array(([[0.5 * (period ** 2)], [period]]))
        if Q_k is None:
            Q_k = lambda k: np.eye(1) * acceleration_variance
    elif method == 1:
        G_k = lambda k: np.eye(2)
        if Q_k is None:
            Q_k = lambda k: acceleration_variance * np.array(
                [[(period ** 4) / 4, (period ** 3) / 2], [(period ** 3) / 2, period ** 2]]
            )
    else:
        raise RuntimeError(f"Got unsupported method '{method}'. Choose from: '0' or '1'.")

    F_k = lambda k: np.array(([[1, period], [0, 1]]))
    C_k = np.array([[1, 0]])
    if R_k is None:
        R_k = np.eye(1) * rng.uniform(0, 1, size=1)
    x_k, r_k = kalman_filter(
        Q_k=Q_k,
        F_k=F_k,
        G_k=G_k,
        C_k=C_k,
        R_k=R_k,
        iterations=iterations,
        rvg_iterations=rvg_iterations,
        m_0=[p0, s0],
        PI_0=np.zeros((2, 2)),
        seed=seed,
    )
    if fig_output_path is not None:
        if Path(fig_output_path).is_dir():
            Path(fig_output_path).mkdir(exist_ok=True, parents=True)
            fig_output_path = Path(fig_output_path) / "kalman_filter.png"
        plot_state_model_vs_observation_roughly_constant_velocity_model(x_k, r_k, fig_output_path)


def main():
    PI_0 = 0  # beginning variance for x(k)
    m_0 = 0  # beginning mean for x(k)
    seed = 1408
    random_vector_generator_iterations = 1000
    # seed = None
    fig_output_path = Path("./kalman_filter.png")
    roughly_constant_velocity_motion_model(
        p0=2,
        s0=3,
        period=1,
        Q_k=None,
        R_k=None,
        rvg_iterations=random_vector_generator_iterations,
        seed=seed,
        fig_output_path=fig_output_path,
        method=0,
    )


if __name__ == "__main__":
    main()
