# /// script
# dependencies = ["numpy", "matplotlib"]
# ///
"""Fisher information matrix of simple policy distributions, drawn as local KL ellipses
in parameter space.

  (a) Gaussian N(mu, sigma^2), theta = (mu, sigma):  F = diag(1/sigma^2, 2/sigma^2).
      Left: the same parameter step dmu = 0.5 is a small change of the distribution when sigma
      is large, and a large change when sigma is small. Right: the same steps in parameter space.
  (b) Mixture 1/2 N(mu1, sigma^2) + 1/2 N(mu2, sigma^2) with fixed sigma, theta = (mu1, mu2).
      No closed form; F is computed by numerical integration. When the components are well
      separated F ~= 1/(2 sigma^2) I, but at mu1 = mu2 it becomes 1/(4 sigma^2) 11^T, which is
      singular: pulling identical components apart does not change the distribution to first order.
Each ellipse is the set of parameter steps with 1/2 dtheta^T F dtheta = delta, i.e. KL ~= delta.
Small ellipse = the policy is very sensitive to its parameters there.

F is checked against the empirical estimate used on the slide,
F ~= 1/N sum_i grad log pi(a_i) grad log pi(a_i)^T.

Run with:  uv run fisher_information.py   (dependencies are declared inline above)
      or:  python fisher_information.py   (with numpy, matplotlib installed)
Writes ../images/fisher-information-gaussian.png (a) and ../images/fisher-information-mixture.png
((b), transparent background), which are stacked in lec04.qmd so that (b) appears on click.
"""
import os

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

rng = np.random.default_rng(0)


def pdf(x, mu, s):
    return np.exp(-0.5 * ((x - mu) / s) ** 2) / (s * np.sqrt(2 * np.pi))


# ------------------------------------------------------------------ Fisher information
SIGMA_MIX = 0.5


def gaussian_fim(mu, sigma):
    return np.diag([1 / sigma**2, 2 / sigma**2])


def mixture_score(x, mu1, mu2, sigma=SIGMA_MIX):
    """d/dmu_k log p(x) = r_k(x) (x - mu_k) / sigma^2, with r_k the responsibility of component k."""
    n1, n2 = pdf(x, mu1, sigma), pdf(x, mu2, sigma)
    return np.stack([n1 * (x - mu1), n2 * (x - mu2)]) / (sigma**2 * (n1 + n2))


def mixture_fim(mu1, mu2, sigma=SIGMA_MIX):
    x = np.linspace(min(mu1, mu2) - 8 * sigma, max(mu1, mu2) + 8 * sigma, 20001)
    p = 0.5 * pdf(x, mu1, sigma) + 0.5 * pdf(x, mu2, sigma)
    score = mixture_score(x, mu1, mu2, sigma)
    return np.einsum("in,jn->ij", score * p * (x[1] - x[0]), score)


def empirical_gaussian_fim(mu, sigma, n=400_000):
    x = rng.normal(mu, sigma, size=n)
    score = np.stack([(x - mu) / sigma**2, ((x - mu) ** 2 - sigma**2) / sigma**3], axis=1)
    return score.T @ score / n


def empirical_mixture_fim(mu1, mu2, sigma=SIGMA_MIX, n=400_000):
    x = np.where(rng.random(n) < 0.5, mu1, mu2) + sigma * rng.normal(size=n)
    score = mixture_score(x, mu1, mu2, sigma)
    return score @ score.T / n


for F, F_hat, name in [(gaussian_fim(0.3, 0.7), empirical_gaussian_fim(0.3, 0.7), "gaussian"),
                       (mixture_fim(-0.4, 0.3), empirical_mixture_fim(-0.4, 0.3), "mixture")]:
    assert np.allclose(F, F_hat, rtol=0.03, atol=0.03), (name, F, F_hat)
    print(f"{name}: F =\n{F}\nempirical F =\n{F_hat}\n")
assert np.allclose(mixture_fim(0.0, 1e-4), np.ones((2, 2)) / (4 * SIGMA_MIX**2), atol=1e-4)
assert np.allclose(mixture_fim(-3.0, 3.0), np.eye(2) / (2 * SIGMA_MIX**2), atol=1e-4)


def kl_ellipse(center, F, delta, n=100):
    """Boundary of {center + d : 1/2 d^T F d = delta}."""
    evals, evecs = np.linalg.eigh(F)
    t = np.linspace(0, 2 * np.pi, n)
    d = evecs @ (np.sqrt(2 * delta / evals)[:, None] * np.stack([np.cos(t), np.sin(t)]))
    return (center[:, None] + d).T


# ------------------------------------------------------------------------- figure
plt.rcParams.update({"font.size": 12, "mathtext.fontset": "cm"})
fig = plt.figure(figsize=(13.5, 5.2), dpi=150)
gs = fig.add_gridspec(2, 3, width_ratios=[1.05, 1.55, 1.1], wspace=0.22, hspace=0.12,
                      left=0.02, right=0.99, top=0.8, bottom=0.1)
ax_a1 = fig.add_subplot(gs[0, 0])
ax_a2 = fig.add_subplot(gs[1, 0], sharex=ax_a1)
ax_b = fig.add_subplot(gs[:, 1])
ax_c = fig.add_subplot(gs[:, 2])

dmu = 0.5
cases = [(1.0, "tab:blue"), (0.3, "tab:orange")]

# (a, left) same parameter step, different change of the distribution
x = np.linspace(-2.8, 3.3, 800)
for ax, (s, color) in zip([ax_a1, ax_a2], cases):
    ax.fill_between(x, pdf(x, 0, s), color=color, alpha=0.2, lw=0)
    ax.plot(x, pdf(x, 0, s), color=color, lw=2, label=r"$\mu = 0$")
    ax.plot(x, pdf(x, dmu, s), color=color, lw=2, ls="--", label=r"$\mu = 0.5$")
    ax.text(0.02, 0.92, rf"$\sigma = {s}$", transform=ax.transAxes, ha="left", va="top", fontsize=13)
    ax.text(0.98, 0.92, rf"KL $= \frac{{\Delta\mu^2}}{{2\sigma^2}} = {dmu**2 / (2 * s**2):.3g}$",
            transform=ax.transAxes, ha="right", va="top", fontsize=13, color=color)
    ax.set_ylim(0, pdf(0, 0, s) * 1.45)
    ax.set_yticks([])
    for side in ["top", "right", "left"]:
        ax.spines[side].set_visible(False)
ax_a1.legend(loc="center right", bbox_to_anchor=(1.0, 0.45), frameon=False, fontsize=11)
plt.setp(ax_a1.get_xticklabels(), visible=False)
ax_a2.set_xlabel(r"action $a$")
ax_a1.set_title("(a) Gaussian policy $\\mathcal{N}(\\mu, \\sigma^2)$, $\\theta = (\\mu, \\sigma)$:   "
                r"$F_\theta = \mathrm{diag}(1/\sigma^2,\ 2/\sigma^2)$" "\n"
                "the same step $\\Delta\\mu = 0.5$ at two values of $\\sigma$, "
                "as densities (left) and in parameter space (right)", loc="left")

# (a, right) Gaussian in (mu, sigma) parameter space
delta_b = 0.012
for sigma in [0.3, 0.65, 1.0, 1.4]:
    for mu in np.arange(-1.0, 1.01, 0.5):
        E = kl_ellipse(np.array([mu, sigma]), gaussian_fim(mu, sigma), delta_b)
        ax_b.add_patch(Polygon(E, closed=True, facecolor="0.55", edgecolor="0.25", alpha=0.35, lw=1))
for s, color in cases:
    E = kl_ellipse(np.array([0.0, s]), gaussian_fim(0.0, s), delta_b)
    ax_b.add_patch(Polygon(E, closed=True, facecolor=color, edgecolor=color, alpha=0.6, lw=1.5))
    ax_b.annotate("", xy=(dmu, s), xytext=(0.0, s), zorder=5,
                  arrowprops=dict(arrowstyle="-|>", color=color, lw=2.2, mutation_scale=14))
for s, color in cases:
    ax_b.text(dmu / 2, s - 0.04 - 0.13 * s, rf"$\Delta\mu = {dmu}$ at $\sigma = {s}$", color=color, ha="center",
              va="top", fontsize=11, zorder=6, bbox=dict(facecolor="white", edgecolor="none", alpha=0.85, pad=1))
ax_b.set_xlim(-1.35, 1.35)
ax_b.set_ylim(0, 1.7)
ax_b.set_aspect("equal")
ax_b.set_xlabel(r"$\mu$")
ax_b.set_ylabel(r"$\sigma$", rotation=0, labelpad=10)
# (b) mixture of two Gaussians in (mu1, mu2) parameter space; mu1 > mu2 is the label-swapped mirror image
delta_c = 0.012
lim = 1.55
mix_cases = [(-0.4, -0.2, "tab:red"), (-0.9, 0.8, "tab:purple")]  # (mu1, mu2, color)
ax_c.plot([-lim, lim], [-lim, lim], color="0.3", ls="--", lw=1.2)
ax_c.text(-0.85, -1.5, r"$\mu_1 = \mu_2$: $F_\theta$ singular", rotation=45, rotation_mode="anchor",
          ha="left", va="bottom", fontsize=11, color="0.25")
for mu1 in np.arange(-1.4, 1.11, 0.5):
    for mu2 in np.arange(-1.2, 1.31, 0.5):
        if mu2 < mu1 or any(np.isclose([mu1, mu2], [m1, m2]).all() for m1, m2, _ in mix_cases):
            continue
        E = kl_ellipse(np.array([mu1, mu2]), mixture_fim(mu1, mu2), delta_c)
        ax_c.add_patch(Polygon(E, closed=True, facecolor="0.55", edgecolor="0.25", alpha=0.35, lw=1))
xs = np.linspace(-2.2, 2.2, 400)
for (mu1, mu2, color), y0 in zip(mix_cases, [-0.6, -1.45]):
    sep = round(mu2 - mu1, 2)
    E = kl_ellipse(np.array([mu1, mu2]), mixture_fim(mu1, mu2), delta_c)
    ax_c.add_patch(Polygon(E, closed=True, facecolor=color, edgecolor=color, alpha=0.6, lw=1.5, zorder=3))
    ins = ax_c.inset_axes([0.45, y0, 1.1, 0.6], transform=ax_c.transData)
    ins.plot(xs, 0.5 * pdf(xs, mu1, SIGMA_MIX), color=color, lw=1, ls="--")
    ins.plot(xs, 0.5 * pdf(xs, mu2, SIGMA_MIX), color=color, lw=1, ls="--")
    mix = 0.5 * pdf(xs, mu1, SIGMA_MIX) + 0.5 * pdf(xs, mu2, SIGMA_MIX)
    ins.fill_between(xs, mix, color=color, alpha=0.2, lw=0)
    ins.plot(xs, mix, color=color, lw=2)
    ins.text(0.02, 0.97, rf"$\mu_2 - \mu_1 = {sep}$", transform=ins.transAxes, ha="left", va="top",
             fontsize=10, color=color)
    ins.set_ylim(0, 0.95)
    ins.set_xticks([])
    ins.set_yticks([])
    for side in ["top", "right", "left"]:
        ins.spines[side].set_visible(False)
ax_c.set_xlim(-lim, lim)
ax_c.set_ylim(-lim, lim)
ax_c.set_aspect("equal")
ax_c.set_xlabel(r"$\mu_1$")
ax_c.set_ylabel(r"$\mu_2$", rotation=0, labelpad=10)
ax_c.set_title("(b) Gaussian mixture, $\\theta = (\\mu_1, \\mu_2)$\n"
               r"$\frac{1}{2}\mathcal{N}(\mu_1, \sigma^2) + \frac{1}{2}\mathcal{N}(\mu_2, \sigma^2)$, "
               rf"$\sigma = {SIGMA_MIX}$", loc="left")
ax_b.set_anchor("N")
ax_c.set_anchor("NW")
for ax in [ax_b, ax_c]:
    for side in ["top", "right"]:
        ax.spines[side].set_visible(False)

caption = fig.text(0.5, 0.975, r"Ellipses: parameter steps $\Delta\theta$ with "
                   r"$\frac{1}{2}\Delta\theta^\top F_\theta\,\Delta\theta = \delta$ $\;(\approx$ KL$(\pi_\theta \,\|\, "
                   r"\pi_{\theta+\Delta\theta}) = \delta)$.   Small ellipse $=$ policy is very sensitive to $\theta$ there.",
                   ha="center", va="top", fontsize=13)

# Two images with identical canvases, stacked on the slide so that (b) appears on click.
out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "images")
ax_c.set_visible(False)
fig.savefig(os.path.join(out_dir, "fisher-information-gaussian.png"), dpi=150, facecolor="white")
for artist in [ax_a1, ax_a2, ax_b, caption]:
    artist.set_visible(False)
ax_c.set_visible(True)
fig.savefig(os.path.join(out_dir, "fisher-information-mixture.png"), dpi=150, transparent=True)
print("saved", out_dir)
