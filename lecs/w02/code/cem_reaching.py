# /// script
# dependencies = ["numpy", "matplotlib", "pillow"]
# ///
"""Cross-Entropy Method (CEM) as an MPC planner on the same 2D point mass / reaching task
used for the MPPI animation.

Each control step runs a few CEM iterations. Each iteration is one frame:
  * sample K action sequences from N(mu, diag(sigma^2))  (faint blue)
  * keep the M lowest-cost 'elites'                     (orange)
  * refit mu, sigma to the elites; the mean plan mu is rolled out (red)
Then the first action of mu is executed (black path), the plan is shifted, and we repeat.

Run with:  uv run cem_reaching.py       (dependencies are declared inline above)
      or:  python cem_reaching.py       (with numpy, matplotlib, pillow installed)
Writes ../images/cem-reaching.gif, which is embedded in lec02.qmd.
"""
import os

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

rng = np.random.default_rng(3)

# ---------------------------------------------------------------- dynamics (same as MPPI)
dt = 0.1
m = 1.0
alpha = 0.5
u_max = 3.0


def step(x, u):
    u = np.clip(u, -u_max, u_max)
    p, v = x[..., :2], x[..., 2:]
    a = (u - alpha * v) / m
    v_new = v + dt * a
    p_new = p + dt * v_new
    return np.concatenate([p_new, v_new], axis=-1)


goal = np.array([3.0, 2.0])
Q_pos, Q_vel, R = 10.0, 0.5, 0.05
QN_pos, QN_vel = 50.0, 5.0


def running_cost(x, u):
    dp = x[..., :2] - goal
    return Q_pos * (dp**2).sum(-1) + Q_vel * (x[..., 2:] ** 2).sum(-1) + R * (u**2).sum(-1)


def terminal_cost(x):
    dp = x[..., :2] - goal
    return QN_pos * (dp**2).sum(-1) + QN_vel * (x[..., 2:] ** 2).sum(-1)


# ---------------------------------------------------------------------- CEM
K = 300  # samples per iteration
M = 30  # elites
N = 25  # horizon
n_iters = 3  # CEM iterations per control step
sigma_init = 1.5  # initial std of the sampling distribution
sigma_min = 0.15  # keep some exploration noise


def rollout(x0, U):
    Kk = U.shape[0]
    X = np.empty((Kk, N + 1, 4))
    X[:, 0] = x0
    S = np.zeros(Kk)
    for t in range(N):
        S += running_cost(X[:, t], U[:, t])
        X[:, t + 1] = step(X[:, t], U[:, t])
    S += terminal_cost(X[:, N])
    return X, S


def cem_iteration(x0, mu, sig):
    U = mu[None] + sig[None] * rng.normal(size=(K, N, 2))
    X, S = rollout(x0, U)
    elite = np.argsort(S)[:M]
    mu_new = U[elite].mean(0)
    sig_new = np.maximum(U[elite].std(0), sigma_min)
    X_mu, _ = rollout(x0, mu_new[None])
    return mu_new, sig_new, X, elite, X_mu[0]


# ------------------------------------------------------------------ simulate
T = 30  # control steps
x = np.array([-2.0, -1.5, 0.0, 0.0])
mu = np.zeros((N, 2))

frames = []
executed = [x[:2].copy()]
for t in range(T):
    sig = np.full((N, 2), sigma_init)  # reset the std every control step, warm start the mean
    for it in range(n_iters):
        mu, sig, X, elite, X_mu = cem_iteration(x, mu, sig)
        frames.append(dict(x=x.copy(), X=X[:, :, :2].copy(), elite=elite.copy(), X_mu=X_mu[:, :2].copy(),
                           path=np.array(executed), step=t, it=it))
    x = step(x, mu[0])
    executed.append(x[:2].copy())
    mu = np.roll(mu, -1, axis=0)
    mu[-1] = 0.0

# ------------------------------------------------------------------- animate
fig, ax = plt.subplots(figsize=(7, 5.2), dpi=110)
ax.set_xlim(-3, 4.2)
ax.set_ylim(-2.5, 3.2)
ax.set_aspect("equal")
ax.set_xticks([])
ax.set_yticks([])
for s in ax.spines.values():
    s.set_visible(False)

ax.add_patch(plt.Circle(goal, 0.18, color="tab:green", zorder=5))
ax.text(goal[0], goal[1] + 0.35, "goal", color="tab:green", fontsize=12, ha="center", va="bottom",
        zorder=7, bbox=dict(facecolor="white", edgecolor="none", alpha=0.8, pad=1.5))

n_show = 120
sample_lines = [ax.plot([], [], color="tab:blue", lw=0.8, alpha=0.12, zorder=1)[0] for _ in range(n_show)]
elite_lines = [ax.plot([], [], color="tab:orange", lw=1.4, alpha=0.7, zorder=2)[0] for _ in range(M)]
mu_line, = ax.plot([], [], color="tab:red", lw=2.5, zorder=4, label="new mean plan $\\mu$ (fit to elites)")
path_line, = ax.plot([], [], color="k", lw=2.0, zorder=3, label="executed path (first action of $\\mu$)")
robot_dot, = ax.plot([], [], "o", color="k", ms=9, zorder=6)
ax.plot([], [], color="tab:blue", lw=1.5, alpha=0.6, label=f"{K} samples from $\\mathcal{{N}}(\\mu, \\Sigma)$")
ax.plot([], [], color="tab:orange", lw=1.8, alpha=0.9, label=f"top {M} elites")
ax.legend(loc="lower right", fontsize=9, frameon=False)
title = ax.set_title("", fontsize=12, loc="left")


def update(i):
    f = frames[i]
    X, elite = f["X"], f["elite"]
    non_elite = np.setdiff1d(np.arange(K), elite)[:n_show]
    for line, k in zip(sample_lines, non_elite):
        line.set_data(X[k, :, 0], X[k, :, 1])
    for line, k in zip(elite_lines, elite):
        line.set_data(X[k, :, 0], X[k, :, 1])
    mu_line.set_data(f["X_mu"][:, 0], f["X_mu"][:, 1])
    path_line.set_data(f["path"][:, 0], f["path"][:, 1])
    robot_dot.set_data([f["x"][0]], [f["x"][1]])
    title.set_text(f"CEM on a 2D point mass, reaching cost      step {f['step'] + 1}/{T},  "
                   f"CEM iteration {f['it'] + 1}/{n_iters}\n"
                   f"K={K} samples, M={M} elites, horizon N={N}")
    return sample_lines + elite_lines + [mu_line, path_line, robot_dot, title]


anim = FuncAnimation(fig, update, frames=len(frames), blit=False)
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "images", "cem-reaching.gif")
anim.save(out, writer=PillowWriter(fps=8))
print("saved", out, len(frames), "frames")
