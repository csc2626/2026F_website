# Website for CSC 2626 at the University of Toronto (Fall 2026)

🔗 https://csc2626.github.io/2026F_website/

## Quarto Website Setup Guide

This guide will help you set up, compile, and preview the CSC2626 course website using Quarto.

### 1. Install Quarto

Visit the official [Quarto website](https://quarto.org/docs/get-started/) to download and install Quarto. Make sure to follow the installation instructions for your operating system (Windows, macOS, Linux). You can verify the installation by running:

``` bash
quarto --version
```

### 2. Clone the Repository

Clone this repository to your local machine:

``` bash
git clone https://github.com/csc2626/2026F_website.git
cd 2026F_website
```

### 3. (Optional) Set Up R

Most of the site — the lecture content under `lecs/` and the course pages (`index.qmd`, etc.) — is plain Quarto markdown with no executable code, and the project uses `freeze: auto` with the results already committed under `_freeze/`. Because of this, a plain `quarto render` works with just Quarto installed; you don't need R or Python for a normal build.

R is only needed if you want to edit and re-execute the computational, revealjs-based slide decks under `slides/` and the R-based lab pages under `labs/lab-*.qmd`, which depend on a large set of packages pinned in `renv.lock` via [renv](https://rstudio.github.io/renv/).

1. Install R (this project was last synced against R 4.4.3, but any recent R 4.x should work).
2. Install a Fortran compiler — some packages (e.g. `rms`, `mvtnorm`) build from source and need one:
   - **macOS**: install the official [R development tools](https://mac.r-project.org/tools/) (provides `gfortran` at `/opt/gfortran`, the path R expects). If you use Homebrew's `gfortran` (`brew install gcc`) instead, tell R where to find it by creating `~/.R/Makevars`:
     ``` make
     FC = /opt/homebrew/bin/gfortran
     F77 = /opt/homebrew/bin/gfortran
     FLIBS = -L/opt/homebrew/lib/gcc/current -lgfortran -lquadmath -lm
     ```
     (adjust the `gcc` lib path to match your `brew --prefix gcc`)
   - **Windows**: install [Rtools](https://cran.r-project.org/bin/windows/Rtools/) matching your R version.
   - **Linux**: install via your package manager, e.g. `sudo apt install gfortran`.
3. From the repo root, install and restore the pinned packages:
   ``` r
   install.packages("renv")
   renv::restore()
   ```
   This installs every package in `renv.lock` at its pinned version, including the two GitHub-only packages (`colorblindr`, `emo`).

> **Note:** the Jupyter notebooks under `labs/` (e.g. `Week4_Experts_Privileged_Information.ipynb`, `Week5_IRL_MaxEnt.ipynb`, the `Week11_*` notebooks) each depend on their own research-specific Python packages (e.g. `gym`, `torch`, `aprel`, and course-specific packages like `ccil`). Quarto publishes their already-saved cell outputs rather than re-executing them, so no Python environment is required to build the site — set one up locally only if you intend to re-run a specific notebook yourself.

### 4. Compile the Website and Run Locally

To compile the source files (.qmd, .md) into a HTML website, run the following command after navigating to the cloned repository:

``` bash
quarto render
```

This will generate the HTML files from the source files and will be placed in the \_site/ directory by default (configurable in \_quarto.yml).

Then to preview the website locally with live reload:

``` bash
quarto preview
```

This command will start a local development server, allowing you to view the website in your browser (e.g. `http://localhost:4200`).

More details on rendering can be found in the [Quarto documentation](https://quarto.org/docs/websites/).

### 5. Deployment

Pushing to `main` automatically triggers the GitHub Actions workflow in [`.github/workflows/build-website.yaml`](.github/workflows/build-website.yaml), which renders the site and publishes the `_site/` directory to the `gh-pages` branch, making it available at `https://csc2626.github.io/2026F_website/`. In most cases you don't need to deploy manually.

If you do need to publish directly from your machine, you can instead run:

``` bash
quarto publish gh-pages
```

This command will push the contents of the `_site/` directory to the `gh-pages` branch of the repository.

## Colors

- website background: #D9E3E4
- headings: #5B888C

## Attribution

Much of the site scaffolding is based on [STA 210 - Fall 2021](https://github.com/sta210-fa21/) by Dr. Maria Tackett and [STA210](https://sta210-s22.github.io/website/) by Dr. Mine Çetinkaya-Rundel.

<hr>

<a rel="license" href="http://creativecommons.org/licenses/by-nc/4.0/"><img src="https://i.creativecommons.org/l/by-nc/4.0/88x31.png" alt="Creative Commons License" style="border-width:0"/></a><br />This work is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by-nc/4.0/">Creative Commons Attribution-NonCommercial 4.0 International License</a>.
