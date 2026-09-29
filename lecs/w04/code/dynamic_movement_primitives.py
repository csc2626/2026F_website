# /// script
# dependencies = ["numpy", "matplotlib"]
# ///
"""Discrete Dynamic Movement Primitives (Ijspeert et al., Neural Computation 2013), one per DoF.

  transformation system:  tau z' = alpha_z (beta_z (g - y) - z) + f(x),     tau y' = z
  canonical system:       tau x' = -alpha_x x,                                x(0) = 1
  forcing term:           f(x) = sum_i psi_i(x) w_i / sum_i psi_i(x) * x (g - y0),
                          psi_i(x) = exp(-h_i (x - c_i)^2)

Learning from one demonstration y_demo(t):
  f_target = tau^2 y_demo'' - alpha_z (beta_z (g - y_demo) - tau y_demo'),   s = x (g - y0)
  w_i = s^T Gamma_i f_target / s^T Gamma_i s,   Gamma_i = diag(psi_i(x_t))   (locally weighted regression)

Figures:
  ../images/dmp-anatomy.png         phase + basis functions, forcing term fit, demo vs reproduction
  ../images/dmp-generalization.png  new goals, temporal scaling, recovery from a perturbation

Run with:  uv run dynamic_movement_primitives.py   (dependencies are declared inline above)
      or:  python dynamic_movement_primitives.py   (with numpy, matplotlib installed)
"""
import os

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ALPHA_Z = 25.0
BETA_Z = ALPHA_Z / 4  # critically damped
ALPHA_X = ALPHA_Z / 3
N_BASIS = 40
DT = 1e-3

# basis centres equally spaced in time, i.e. exponentially spaced in the phase x
C = np.exp(-ALPHA_X * np.linspace(0, 1, N_BASIS))
H = 1.0 / np.diff(C) ** 2
H = np.append(H, H[-1])


def psi(x):
    return np.exp(-H * (np.asarray(x)[..., None] - C) ** 2)


def demo(t):
    """2D demonstration of duration 1: a minimum-jerk reach with an S-shaped detour."""
    s = 10 * t**3 - 15 * t**4 + 6 * t**5
    return np.stack([s + 0.12 * np.sin(2 * np.pi * s), 0.5 * s + 0.35 * np.sin(2 * np.pi * s)], axis=-1)


def fit(y_demo, tau):
    y0, g = y_demo[0], y_demo[-1]
    yd = np.gradient(y_demo, DT, axis=0)
    ydd = np.gradient(yd, DT, axis=0)
    t = np.arange(len(y_demo)) * DT
    x = np.exp(-ALPHA_X * t / tau)
    f_target = tau**2 * ydd - ALPHA_Z * (BETA_Z * (g - y_demo) - tau * yd)
    P = psi(x)  # (T, N_BASIS)
    w = np.zeros((N_BASIS, y_demo.shape[1]))
    for d in range(y_demo.shape[1]):
        s = x * (g[d] - y0[d])
        w[:, d] = (P * (s * f_target[:, d])[:, None]).sum(0) / (P * (s**2)[:, None]).sum(0)
    return w, x, f_target


def forcing(w, x, y0, g):
    P = psi(x)
    return (P @ w) / P.sum(-1, keepdims=True) * (x[..., None] * (g - y0))


def rollout(w, y0, g, tau, T, push=None):
    """Integrate the DMP for T seconds. push = (time, displacement) instantaneously displaces y."""
    n = int(round(T / DT))
    y, z, x = y0.astype(float).copy(), np.zeros_like(y0, dtype=float), 1.0
    ys, xs = [y.copy()], [x]
    for k in range(n):
        if push is not None and k == int(round(push[0] / DT)):
            y = y + push[1]
        f = forcing(w, np.array(x), y0, g)
        zd = (ALPHA_Z * (BETA_Z * (g - y) - z) + f) / tau
        yd = z / tau
        xd = -ALPHA_X * x / tau
        z, y, x = z + DT * zd, y + DT * yd, x + DT * xd
        ys.append(y.copy())
        xs.append(x)
    return np.array(ys), np.array(xs)


TAU = 1.0
t_demo = np.arange(0, TAU + DT / 2, DT)
y_demo = demo(t_demo)
y0, g = y_demo[0], y_demo[-1]
w, x_demo, f_target = fit(y_demo, TAU)
y_rep, _ = rollout(w, y0, g, TAU, 1.5 * TAU)
assert np.abs(y_rep[: len(y_demo)] - y_demo).max() < 0.02
assert np.abs(y_rep[-1] - g).max() < 1e-3

plt.rcParams.update({"font.size": 15, "mathtext.fontset": "cm"})
DEMO_KW = dict(color="k", lw=5, alpha=0.18, solid_capstyle="round")


def clean(ax):
    for side in ["top", "right"]:
        ax.spines[side].set_visible(False)


def mark_ends(ax, y0, g, color="k"):
    ax.plot(*y0, "o", color="k", ms=7, zorder=6)
    ax.plot(*g, "*", color=color, ms=16, zorder=6, mec="k", mew=0.6)


# ------------------------------------------------------------------------- anatomy figure
fig, (ax_a, ax_b, ax_c) = plt.subplots(1, 3, figsize=(15, 4.3), dpi=150, gridspec_kw=dict(width_ratios=[1, 1, 0.9]))
fig.subplots_adjust(left=0.05, right=0.99, top=0.84, bottom=0.15, wspace=0.28)

# (a) phase variable and the basis functions it activates over time
P = psi(x_demo)
cmap = plt.get_cmap("viridis")
for i in range(0, N_BASIS, 3):
    ax_a.plot(t_demo, P[:, i], color=cmap(i / (N_BASIS - 1)), lw=1.4)
ax_a.plot(t_demo, x_demo, "k--", lw=2.2)
ax_a.text(0.1, 0.45, r"phase $x(t)$", fontsize=16, bbox=dict(facecolor="white", edgecolor="none", alpha=0.9, pad=1))
ax_a.set_xlabel(r"time $t$")
ax_a.set_title(r"Canonical system $x(t) = e^{-\alpha_x t/\tau}$" "\n" r"and basis functions $\psi_i(x(t))$ (every 3rd)",
               loc="left", fontsize=15)
ax_a.set_ylim(0, 1.05)
clean(ax_a)

# (b) forcing term: target from the demo vs weighted sum of basis functions (2nd DoF)
d = 1
s = x_demo * (g[d] - y0[d])
contrib = P * w[:, d] / P.sum(-1, keepdims=True) * s[:, None]
for i in range(N_BASIS):
    ax_b.plot(t_demo, contrib[:, i], color=cmap(i / (N_BASIS - 1)), lw=1, alpha=0.8)
ax_b.plot(t_demo, f_target[:, d], color="k", lw=5, alpha=0.18, label=r"$f_{\mathrm{target}}$ from demo")
ax_b.plot(t_demo, contrib.sum(1), color="tab:red", lw=2, label=r"learned $f(x)$")
ax_b.axhline(0, color="0.6", lw=0.8)
ax_b.set_xlabel(r"time $t$")
ax_b.set_title("Forcing term for $y_2$: sum of weighted,\n" r"normalized basis functions $\times\, x\,(g - y_0)$",
               loc="left", fontsize=15)
ax_b.legend(frameon=False, fontsize=13, loc="lower right")
clean(ax_b)

# (c) demonstration vs DMP rollout
ax_c.plot(y_demo[:, 0], y_demo[:, 1], **DEMO_KW)
ax_c.plot(y_rep[:, 0], y_rep[:, 1], color="tab:red", lw=2)
mark_ends(ax_c, y0, g, "tab:red")
ax_c.text(y0[0] + 0.05, y0[1] + 0.01, r"$y_0$", fontsize=17, va="bottom")
ax_c.text(g[0] - 0.03, g[1] + 0.02, r"$g$", fontsize=17, ha="right", va="bottom")
ax_c.text(-0.02, 0.6, "demonstration", color="0.55", fontsize=13)
ax_c.text(-0.02, 0.53, "DMP rollout", color="tab:red", fontsize=13)
ax_c.set_xlabel(r"$y_1$")
ax_c.set_ylabel(r"$y_2$", rotation=0, labelpad=10)
ax_c.set_xlim(-0.08, 1.08)
ax_c.set_ylim(-0.08, 0.67)
ax_c.set_aspect("equal")
ax_c.set_title(f"One demonstration,\n{N_BASIS} weights per DoF", loc="left", fontsize=15)
clean(ax_c)

here = os.path.dirname(os.path.abspath(__file__))
out = os.path.join(here, "..", "images", "dmp-anatomy.png")
fig.savefig(out, dpi=150, facecolor="white")
print("saved", out)

# ------------------------------------------------------------------------- generalization figure
fig, (ax_a, ax_b, ax_c) = plt.subplots(1, 3, figsize=(15, 4.3), dpi=150, gridspec_kw=dict(width_ratios=[1, 1.1, 1]))
fig.subplots_adjust(left=0.05, right=0.99, top=0.84, bottom=0.15, wspace=0.25)

# (a) same weights, new goals
ax_a.plot(y_demo[:, 0], y_demo[:, 1], **DEMO_KW)
goals = [np.array([1.3, 0.9]), np.array([0.7, 0.8]), np.array([1.25, 0.2])]
colors = ["tab:blue", "tab:green", "tab:purple"]
for g_new, color in zip(goals, colors):
    ys, _ = rollout(w, y0, g_new, TAU, 1.5 * TAU)
    ax_a.plot(ys[:, 0], ys[:, 1], color=color, lw=2)
    mark_ends(ax_a, y0, g_new, color)
mark_ends(ax_a, y0, g, "0.7")
ax_a.set_xlabel(r"$y_1$")
ax_a.set_ylabel(r"$y_2$", rotation=0, labelpad=10)
ax_a.set_aspect("equal")
ax_a.set_title("Spatial scaling: change $g$,\nkeep the weights $w$", loc="left", fontsize=15)
clean(ax_a)

# (b) same weights, different tau
for tau, color in [(0.5, "tab:orange"), (1.0, "tab:red"), (2.0, "tab:brown")]:
    ys, _ = rollout(w, y0, g, tau, 2.6)
    ax_b.plot(np.arange(len(ys)) * DT, ys[:, 1], color=color, lw=2, label=rf"$\tau = {tau:g}$")
ax_b.plot(t_demo, y_demo[:, 1], **DEMO_KW)
ax_b.legend(frameon=False, fontsize=14, loc="lower right")
ax_b.axhline(g[1], color="0.6", lw=0.8, ls=":")
ax_b.set_xlabel(r"time $t$")
ax_b.set_ylabel(r"$y_2$", rotation=0, labelpad=10)
ax_b.set_title(r"Temporal scaling: change $\tau$," "\n" "same path, different speed", loc="left", fontsize=15)
clean(ax_b)

# (c) push the system mid-movement
ax_c.plot(y_demo[:, 0], y_demo[:, 1], **DEMO_KW)
t_push, push = 0.35, np.array([0.12, -0.3])
ys, _ = rollout(w, y0, g, TAU, 1.5 * TAU, push=(t_push, push))
k = int(round(t_push / DT))
ax_c.plot(ys[: k + 1, 0], ys[: k + 1, 1], color="tab:red", lw=2)
ax_c.plot(ys[k + 1 :, 0], ys[k + 1 :, 1], color="tab:red", lw=2)
ax_c.annotate("", xy=ys[k + 1], xytext=ys[k], arrowprops=dict(arrowstyle="-|>", color="k", lw=2, mutation_scale=18))
ax_c.text(*(ys[k] + push / 2 + np.array([0.04, 0.0])), "push", fontsize=15, ha="left", va="center")
mark_ends(ax_c, y0, g, "tab:red")
ax_c.set_xlabel(r"$y_1$")
ax_c.set_ylabel(r"$y_2$", rotation=0, labelpad=10)
ax_c.set_aspect("equal")
ax_c.set_title("Perturbation at $t = 0.35$:\nno replanning, still reaches $g$", loc="left", fontsize=15)
clean(ax_c)

out = os.path.join(here, "..", "images", "dmp-generalization.png")
fig.savefig(out, dpi=150, facecolor="white")
print("saved", out)
