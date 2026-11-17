# /// script
# dependencies = ["numpy", "matplotlib"]
# ///
"""Illustrations of the stability definitions for a closed-loop system x' = f_cl(x), equilibrium x* = 0.

Panels, in a 2x2 grid read row by row:

  1. Stable (Lyapunov), not asymptotically:  undamped oscillator  x1' = x2,  x2' = -x1.
     A trajectory starting in the delta-ball circles forever, inside the epsilon-ball, and never settles.
  2. Asymptotically stable:                  damped oscillator    x1' = x2,  x2' = -x1 - 0.5 x2.
     Every trajectory spirals into x*.
  3. Exponential vs only asymptotic:         ||x(t)|| on a log axis for the scalar systems
         x' = -x     ->  x(t) = x0 e^{-t}                       (a straight line, below c e^{-lam t})
         x' = -x^3   ->  x(t) = x0 / sqrt(1 + 2 x0^2 t)         (decays like 1/sqrt(t), so it ends up
                                                                  above every exponential envelope)
  4. Region of attraction:                   Van der Pol in reverse time
         x1' = -x2,  x2' = x1 - (1 - x1^2) x2.
     The origin is asymptotically stable only locally. The region of attraction is the inside of the
     unstable limit cycle (the stable limit cycle of the forward-time Van der Pol oscillator).

Run with:  uv run stability_definitions.py   (dependencies are declared inline above)
      or:  python stability_definitions.py   (with numpy, matplotlib installed)
Writes ../images/stability-definitions.png.
"""
import os

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Polygon
from matplotlib.path import Path

TEAL, RED, GREY = "#2a9fbc", "#e4473a", "0.6"


def rk4(f, x0, dt, n_steps, max_norm=np.inf):
    xs = [np.asarray(x0, dtype=float)]
    for _ in range(n_steps):
        x = xs[-1]
        k1 = f(x)
        k2 = f(x + dt / 2 * k1)
        k3 = f(x + dt / 2 * k2)
        k4 = f(x + dt * k3)
        xs.append(x + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4))
        if np.linalg.norm(xs[-1]) > max_norm:
            break
    return np.array(xs)


def undamped(x):
    return np.array([x[1], -x[0]])


def damped(x):
    return np.array([x[1], -x[0] - 0.5 * x[1]])


def vdp_reverse(x):
    return np.array([-x[1], x[0] - (1 - x[0] ** 2) * x[1]])


def vdp_forward(x):
    return -vdp_reverse(x)


def stream(ax, f, lim):
    g = np.linspace(-lim, lim, 25)
    X1, X2 = np.meshgrid(g, g)
    U, V = f(np.stack([X1, X2]))
    ax.streamplot(X1, X2, U, V, color=GREY, linewidth=0.6, density=0.8, arrowsize=0.7)


def mark_equilibrium(ax):
    ax.plot(0, 0, "ko", ms=5, zorder=5)
    ax.annotate(r"$x^*$", (0, 0), xytext=(6, -12), textcoords="offset points", fontsize=11)


def phase_axes(ax, lim, title):
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel(r"$x_1$")
    ax.set_ylabel(r"$x_2$")
    ax.set_title(title, fontsize=12)


fig, axes = plt.subplots(2, 2, figsize=(8.4, 7.6))
axes = axes.ravel()

# 1. Lyapunov stable, not asymptotically stable.
ax = axes[0]
delta, eps = 0.75, 1.25
stream(ax, undamped, 1.6)
ax.add_patch(Circle((0, 0), eps, fill=False, ls="--", color=RED, lw=1.5))
ax.add_patch(Circle((0, 0), delta, fill=False, ls="--", color=TEAL, lw=1.5))
ax.text(0.80 * eps, 0.80 * eps, r"$\epsilon$", color=RED, fontsize=13)
ax.text(0.45 * delta, 0.62 * delta, r"$\delta$", color=TEAL, fontsize=13)
x0 = np.array([0.38, 0.25])
traj = rk4(undamped, x0, 0.01, 700)
ax.plot(traj[:, 0], traj[:, 1], color=TEAL, lw=2.2)
ax.plot(*x0, "o", color=TEAL, ms=6)
mark_equilibrium(ax)
phase_axes(ax, 1.6, "Stable: start close, stay close")

# 2. Asymptotically stable.
ax = axes[1]
stream(ax, damped, 1.6)
for x0 in [(1.3, 0.6), (-1.2, -0.9), (-0.4, 1.4)]:
    traj = rk4(damped, x0, 0.01, 3000)
    ax.plot(traj[:, 0], traj[:, 1], color=TEAL, lw=2.0)
    ax.plot(*x0, "o", color=TEAL, ms=6)
mark_equilibrium(ax)
phase_axes(ax, 1.6, r"Asymptotically stable: $x(t) \to x^*$")

# 3. Exponential vs only asymptotic convergence.
ax = axes[2]
t = np.linspace(0, 12, 400)
c, lam = 1.3, 0.35
ax.semilogy(t, c * np.exp(-lam * t), "--", color="k", lw=1.3, label=r"envelope $c\,e^{-\lambda t}$")
ax.semilogy(t, np.exp(-t), color=TEAL, lw=2.2, label=r"$\dot x = -x$: exponential")
ax.semilogy(t, 1 / np.sqrt(1 + 2 * t), color=RED, lw=2.2, label=r"$\dot x = -x^3$: only asymptotic")
ax.set_ylim(1e-4, 2)
ax.set_xlim(0, 12)
ax.set_xlabel(r"$t$")
ax.set_ylabel(r"$\|x(t) - x^*\|$")
ax.legend(fontsize=9.5, loc="lower left", frameon=False)
ax.set_title("Exponentially stable", fontsize=12)
ax.spines[["top", "right"]].set_visible(False)

# 4. Region of attraction of the reverse-time Van der Pol oscillator.
ax = axes[3]
lim = 3.2
transient = rk4(vdp_forward, (2.0, 0.0), 0.005, 8000)
cycle = rk4(vdp_forward, transient[-1], 0.005, 1400)  # one period is about 6.66 time units
stream(ax, vdp_reverse, lim)
ax.add_patch(Polygon(cycle, closed=True, facecolor=TEAL, alpha=0.18, edgecolor="none"))
ax.plot(cycle[:, 0], cycle[:, 1], color="k", lw=1.6)
ax.text(-0.4, -2.95, "unstable limit cycle", fontsize=10)
inside, outside = np.array([1.2, 1.0]), np.array([1.6, 2.3])
roa = Path(cycle)
assert roa.contains_point(inside) and not roa.contains_point(outside)
traj = rk4(vdp_reverse, inside, 0.005, 6000)
ax.plot(traj[:, 0], traj[:, 1], color=TEAL, lw=2.0)
ax.plot(*inside, "o", color=TEAL, ms=6)
traj = rk4(vdp_reverse, outside, 0.002, 6000, max_norm=2 * lim)
ax.plot(traj[:, 0], traj[:, 1], color=RED, lw=2.0)
ax.plot(*outside, "o", color=RED, ms=6)
ax.text(-2.2, -0.5, "region of\nattraction", fontsize=10, color="#14627a")
mark_equilibrium(ax)
phase_axes(ax, lim, "Region of attraction")

fig.tight_layout()
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "images", "stability-definitions.png")
fig.savefig(out, dpi=250, bbox_inches="tight")
print("wrote", os.path.normpath(out))
