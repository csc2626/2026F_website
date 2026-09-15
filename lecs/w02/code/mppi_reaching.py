# /// script
# dependencies = ["numpy", "matplotlib", "pillow"]
# ///
"""MPPI on a 2D point mass (double integrator) with a quadratic reaching cost.

Produces an animated GIF showing, at every control step:
  * the K sampled rollouts, coloured/alpha-weighted by their softmin weight w_k
  * the updated nominal plan (weighted average of the samples)
  * the path executed so far (MPC: only the first action is applied)

Run with:  uv run mppi_reaching.py      (dependencies are declared inline above)
      or:  python mppi_reaching.py      (with numpy, matplotlib, pillow installed)
Writes ../images/mppi-reaching.gif, which is embedded in lec02.qmd.
"""
import os

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

rng = np.random.default_rng(3)

# ---------------------------------------------------------------- dynamics
dt = 0.1
m = 1.0
alpha = 0.5  # friction
u_max = 3.0


def step(x, u):
    """x: (..., 4) = [px, py, vx, vy], u: (..., 2) force. Vectorised over leading dims."""
    u = np.clip(u, -u_max, u_max)
    p, v = x[..., :2], x[..., 2:]
    a = (u - alpha * v) / m
    v_new = v + dt * a
    p_new = p + dt * v_new
    return np.concatenate([p_new, v_new], axis=-1)


# --------------------------------------------------------------------- cost
goal = np.array([3.0, 2.0])
Q_pos, Q_vel, R = 10.0, 0.5, 0.05
QN_pos, QN_vel = 50.0, 5.0


def running_cost(x, u):
    dp = x[..., :2] - goal
    return Q_pos * (dp**2).sum(-1) + Q_vel * (x[..., 2:] ** 2).sum(-1) + R * (u**2).sum(-1)


def terminal_cost(x):
    dp = x[..., :2] - goal
    return QN_pos * (dp**2).sum(-1) + QN_vel * (x[..., 2:] ** 2).sum(-1)


# --------------------------------------------------------------------- MPPI
K = 300  # samples
N = 25  # horizon
lam = 80.0  # temperature
sigma = 1.5  # exploration std on the force


def rollout(x0, U):
    """U: (K, N, 2) -> states (K, N+1, 4), costs (K,)"""
    Kk = U.shape[0]
    X = np.empty((Kk, N + 1, 4))
    X[:, 0] = x0
    S = np.zeros(Kk)
    for t in range(N):
        S += running_cost(X[:, t], U[:, t])
        X[:, t + 1] = step(X[:, t], U[:, t])
    S += terminal_cost(X[:, N])
    return X, S


def mppi_update(x0, U_nom):
    eps = rng.normal(0.0, sigma, size=(K, N, 2))
    U = U_nom[None] + eps
    X, S = rollout(x0, U)
    w = np.exp(-(S - S.min()) / lam)
    w /= w.sum()
    U_new = U_nom + np.einsum("k,ktd->td", w, eps)
    X_nom, _ = rollout(x0, U_new[None])
    return U_new, X, w, X_nom[0]


# ---------------------------------------------------------------- simulate
T = 36  # control steps in the animation
x = np.array([-2.0, -1.5, 0.0, 0.0])
U_nom = np.zeros((N, 2))

frames = []
executed = [x[:2].copy()]
for t in range(T):
    U_nom, X, w, X_nom = mppi_update(x, U_nom)
    frames.append(dict(x=x.copy(), X=X[:, :, :2].copy(), w=w.copy(), X_nom=X_nom[:, :2].copy(),
                       path=np.array(executed)))
    x = step(x, U_nom[0])  # execute first action
    executed.append(x[:2].copy())
    U_nom = np.roll(U_nom, -1, axis=0)  # shift plan, warm start
    U_nom[-1] = 0.0

# ----------------------------------------------------------------- animate
fig, ax = plt.subplots(figsize=(7, 5.2), dpi=110)
ax.set_xlim(-3, 4.2)
ax.set_ylim(-2.5, 3.2)
ax.set_aspect("equal")
ax.set_xticks([])
ax.set_yticks([])
for s in ax.spines.values():
    s.set_visible(False)

goal_circ = plt.Circle(goal, 0.18, color="tab:green", zorder=5)
ax.add_patch(goal_circ)
ax.text(goal[0], goal[1] + 0.35, "goal", color="tab:green", fontsize=12, ha="center", va="bottom",
        zorder=7, bbox=dict(facecolor="white", edgecolor="none", alpha=0.8, pad=1.5))

n_show = 120  # subset of samples to draw
sample_lines = [ax.plot([], [], color="tab:blue", lw=1.0, alpha=0.1, zorder=1)[0] for _ in range(n_show)]
nom_line, = ax.plot([], [], color="tab:red", lw=2.5, zorder=4, label="updated nominal plan  $\\bar{u} + \\sum_k w_k \\epsilon^k$")
path_line, = ax.plot([], [], color="k", lw=2.0, zorder=3, label="executed path (first action only)")
robot_dot, = ax.plot([], [], "o", color="k", ms=9, zorder=6)
proxy = ax.plot([], [], color="tab:blue", lw=1.5, alpha=0.6, label=f"{K} sampled rollouts (opacity $\\propto w_k$)")[0]
ax.legend(loc="lower right", fontsize=9, frameon=False)
title = ax.set_title("", fontsize=12, loc="left")

show_idx = np.arange(n_show)


def update(i):
    f = frames[i]
    X, w = f["X"], f["w"]
    # draw a fixed subset, but always include the highest-weight samples
    top = np.argsort(w)[::-1][: n_show // 3]
    idx = np.unique(np.concatenate([top, show_idx]))[:n_show]
    wmax = w.max()
    idx = idx[np.argsort(w[idx])]  # draw high-weight samples last (on top)
    for line, k in zip(sample_lines, idx):
        r = (w[k] / wmax) ** 0.75
        line.set_data(X[k, :, 0], X[k, :, 1])
        line.set_alpha(float(np.clip(0.06 + 0.85 * r, 0.06, 0.95)))
        line.set_linewidth(0.7 + 2.0 * r)
    for line in sample_lines[len(idx):]:
        line.set_data([], [])
    nom_line.set_data(f["X_nom"][:, 0], f["X_nom"][:, 1])
    path_line.set_data(f["path"][:, 0], f["path"][:, 1])
    robot_dot.set_data([f["x"][0]], [f["x"][1]])
    title.set_text(f"MPPI on a 2D point mass, reaching cost      step {i + 1}/{T}\n"
                   f"K={K} samples, horizon N={N}, $\\lambda$={lam:g}, $\\sigma$={sigma:g}")
    return sample_lines + [nom_line, path_line, robot_dot, title]


anim = FuncAnimation(fig, update, frames=len(frames), blit=False)
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "images", "mppi-reaching.gif")
anim.save(out, writer=PillowWriter(fps=6))
print("saved", out)
