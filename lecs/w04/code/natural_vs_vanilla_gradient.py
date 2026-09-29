# /// script
# dependencies = ["numpy", "matplotlib"]
# ///
"""Vanilla vs natural gradient descent on a loss that is much more sensitive in one direction.

Model p_theta(x) = N(x; theta, Sigma) in 2D with a fixed, elongated covariance Sigma. Fitting theta
to data by maximum likelihood gives the loss
    L(theta) = E_data[-log p_theta(x)] = 1/2 (theta - xbar)^T Sigma^{-1} (theta - xbar) + const,
    grad L = Sigma^{-1} (theta - xbar),    F_theta = Sigma^{-1}.
The eigenvalues of Sigma^{-1} are LAM_STIFF >> LAM_FLAT, so L is a narrow valley.

  vanilla:  theta <- theta - alpha grad L            stable only for alpha < 2 / LAM_STIFF, so the flat
                                                     direction shrinks by at best (k - 1)/(k + 1) per step
  natural:  theta <- theta - alpha F^{-1} grad L     = theta - alpha (theta - xbar), a straight line to
                                                     xbar that shrinks by (1 - alpha) per step for any k
with k = LAM_STIFF / LAM_FLAT the condition number.

Left: trajectories on the loss contours. Around the start, the dashed circle is the Euclidean ball
(vanilla = steepest descent inside it) and the ellipse is the KL ball 1/2 d^T F d = delta
(natural = steepest descent inside it). Right: loss vs iteration.

Run with:  uv run natural_vs_vanilla_gradient.py   (dependencies are declared inline above)
      or:  python natural_vs_vanilla_gradient.py   (with numpy, matplotlib installed)
Writes ../images/natural-vs-vanilla-gradient.png.
"""
import os

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Polygon

LAM_STIFF, LAM_FLAT = 25.0, 1.0
PHI = np.deg2rad(20)  # direction of the flat valley floor
v_flat = np.array([np.cos(PHI), np.sin(PHI)])
v_stiff = np.array([-np.sin(PHI), np.cos(PHI)])
R = np.stack([v_stiff, v_flat], axis=1)
F = R @ np.diag([LAM_STIFF, LAM_FLAT]) @ R.T  # = Sigma^{-1}, also the Hessian of L
xbar = np.zeros(2)
theta0 = -2.6 * v_flat + 0.35 * v_stiff


def loss(theta):
    d = theta - xbar
    return 0.5 * np.einsum("...i,ij,...j->...", d, F, d)


def grad(theta):
    return F @ (theta - xbar)


def run(alpha, natural, n_steps):
    thetas = [theta0]
    for _ in range(n_steps):
        g = grad(thetas[-1])
        step = np.linalg.solve(F, g) if natural else g
        thetas.append(thetas[-1] - alpha * step)
    return np.array(thetas)


def kl_ellipse(center, F, delta, n=200):
    """Boundary of {center + d : 1/2 d^T F d = delta}."""
    evals, evecs = np.linalg.eigh(F)
    t = np.linspace(0, 2 * np.pi, n)
    d = evecs @ (np.sqrt(2 * delta / evals)[:, None] * np.stack([np.cos(t), np.sin(t)]))
    return (center[:, None] + d).T


N_STEPS = 40
alpha_best = 2 / (LAM_STIFF + LAM_FLAT)
runs = [  # (thetas, color, label)
    (run(alpha_best, False, N_STEPS), "tab:orange", rf"vanilla, best fixed $\alpha = {alpha_best:.3f}$"),
    (run(0.4 / LAM_STIFF, False, N_STEPS), "tab:red", rf"vanilla, small $\alpha = {0.4 / LAM_STIFF:.3f}$"),
    (run(0.25, True, N_STEPS), "tab:blue", "natural\n" r"$\alpha = 0.25$"),
]

kappa = LAM_STIFF / LAM_FLAT
th = runs[0][0]
flat_err = (th - xbar) @ v_flat
assert np.allclose(flat_err[1:] / flat_err[:-1], (kappa - 1) / (kappa + 1))
th = runs[2][0]
d, d0 = th - xbar, theta0 - xbar
assert np.allclose(d[:, 0] * d0[1] - d[:, 1] * d0[0], 0, atol=1e-12)  # natural path is a straight line
assert np.allclose(loss(th[1:]) / loss(th[:-1]), 0.75**2)

# ------------------------------------------------------------------------- figure
plt.rcParams.update({"font.size": 15, "mathtext.fontset": "cm"})
fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(12.5, 4.6), dpi=150, gridspec_kw=dict(width_ratios=[1.65, 1]))
fig.subplots_adjust(left=0.03, right=0.99, top=0.84, bottom=0.12, wspace=0.22)

# (a) trajectories on the loss contours
xs = np.linspace(-3.25, 0.85, 400)
ys = np.linspace(-1.3, 0.6, 300)
X, Y = np.meshgrid(xs, ys)
Z = loss(np.stack([X, Y], axis=-1))
levels = np.array([0.01, 0.04, 0.12, 0.3, 0.6, 1.1, 1.8, 2.8, 4, 5.5])
ax_a.contourf(X, Y, Z, levels=np.concatenate([[0], levels, [Z.max()]]), cmap="Greys", alpha=0.18)
ax_a.contour(X, Y, Z, levels=levels, colors="0.55", linewidths=0.8)

for thetas, color, label in runs:
    ax_a.plot(thetas[:, 0], thetas[:, 1], "-o", color=color, lw=1.6, ms=3, zorder=4)
ax_a.plot(*xbar, marker="*", ms=16, color="k", zorder=6)
ax_a.text(xbar[0] + 0.1, xbar[1] - 0.08, r"$\theta^\star = \bar{x}$", fontsize=16, va="top")
ax_a.plot(*theta0, "o", ms=7, color="k", zorder=6)
ax_a.text(theta0[0] - 0.38, theta0[1], r"$\theta_0$", fontsize=17, ha="right", va="center")

# one step from theta0: steepest descent inside a Euclidean ball vs inside a KL ball
g0 = grad(theta0)
r = 0.33
ax_a.add_patch(Circle(theta0, r, facecolor="none", edgecolor="tab:orange", ls="--", lw=1.5, zorder=5))
d_van = -r * g0 / np.linalg.norm(g0)
delta = 0.5 * 0.55**2 * LAM_FLAT
ax_a.add_patch(Polygon(kl_ellipse(theta0, F, delta), closed=True, facecolor="tab:blue", alpha=0.2,
                       edgecolor="tab:blue", lw=1.5, zorder=5))
d_nat = -np.linalg.solve(F, g0)
d_nat *= np.sqrt(2 * delta / (d_nat @ F @ d_nat))
for d, color in [(d_van, "tab:orange"), (d_nat, "tab:blue")]:
    ax_a.annotate("", xy=theta0 + d, xytext=theta0, zorder=7,
                  arrowprops=dict(arrowstyle="-|>", color=color, lw=2.8, mutation_scale=18))
label_box = dict(facecolor="white", edgecolor="none", alpha=0.85, pad=1)
ax_a.text(theta0[0] - 0.3, theta0[1] - 0.3, r"$-\nabla L$", color="tab:orange", fontsize=17,
          ha="right", va="top", zorder=7)
ax_a.text(theta0[0] + 0.3, theta0[1] + 0.4, r"$-F^{-1}\nabla L$", color="tab:blue", fontsize=17,
          ha="center", va="bottom", zorder=7)

# sensitive vs insensitive directions
c = np.array([-0.5, -1.2])
for v, ln in [(v_stiff, 0.3), (v_flat, 0.6)]:
    ax_a.annotate("", xy=c + ln * v, xytext=c, arrowprops=dict(arrowstyle="-|>", color="0.2", lw=1.5))
ax_a.text(*(c + 0.3 * v_stiff + np.array([-0.06, 0.0])), "sensitive:\n" rf"$\lambda_{{\max}}(F) = {LAM_STIFF:g}$",
          fontsize=14, ha="right", va="center", color="0.2", bbox=label_box)
ax_a.text(*(c + 0.6 * v_flat + np.array([0.05, 0.0])), "insensitive:\n" rf"$\lambda_{{\min}}(F) = {LAM_FLAT:g}$",
          fontsize=14, ha="left", va="center", color="0.2", bbox=label_box)

ax_a.set_xlim(xs[0], xs[-1])
ax_a.set_ylim(ys[0], ys[-1])
ax_a.set_aspect("equal")
ax_a.set_xticks([])
ax_a.set_yticks([])
ax_a.set_xlabel(r"$\theta_1$")
ax_a.set_ylabel(r"$\theta_2$", rotation=0, labelpad=10)
ax_a.set_title(r"Contours of $L(\theta)$" "\n" r"dashed circle: $\|\Delta\theta\| = r$,   "
               r"shaded ellipse: $\frac{1}{2}\Delta\theta^\top F_\theta\,\Delta\theta = \delta$", loc="left",
               fontsize=15)

# (b) loss vs iteration
for (thetas, color, label), (k, y, ha, va) in zip(runs, [(40, 1e-3, "right", "center"),
                                                         (40, 1.6, "right", "bottom"),
                                                         (13, 3e-6, "right", "center")]):
    ax_b.semilogy(loss(thetas) - loss(xbar), "-o", color=color, lw=1.6, ms=3)
    ax_b.text(k, y, label, color=color, fontsize=14.5, ha=ha, va=va)
ax_b.set_xlabel("iteration $k$")
ax_b.set_ylabel(r"$L(\theta_k) - L(\theta^\star)$")
ax_b.set_xlim(0, N_STEPS)
ax_b.set_ylim(1e-8, 10)
ax_b.set_title("Loss vs iteration\n" rf"condition number $\kappa = \lambda_{{\max}} / \lambda_{{\min}} = {kappa:g}$", loc="left",
               fontsize=15)
for side in ["top", "right"]:
    ax_b.spines[side].set_visible(False)

out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "images", "natural-vs-vanilla-gradient.png")
fig.savefig(out, dpi=150, facecolor="white")
print("saved", out)

