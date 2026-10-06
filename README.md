# CSI ODE Experiments

Formal experimental code, recorded results, and figure reproduction for the
conditional ODE study with low-rank pairwise interactions.

## Contents

- `experiments/code/`: four neural particle-field experiments and the
  variance-preserving endpoint-CSI conditional-feature (CF) experiment.
- `experiments/data/`: five formal result sets, with the recorded configurations,
  per-run results, restart results, and summaries.
- `experiments/provenance/`: data and delivery checksums, source records, and
  available historical environment information.
- `figures/`: the six PDF figures used in the manuscript.

The experimental package is kept in an `experiments/` subdirectory so its
frozen manifests and relative paths remain unchanged. In its reproduction
instructions, "paper directory" means this repository's root directory.

## Verify results and reproduce figures

Run these commands from the repository root in a separate Python environment:

```sh
python3 -m pip install -r experiments/requirements-figures.txt
python3 -B experiments/verify_results.py
python3 -B experiments/plot_figures.py
```

Verification does not run training. The plotter redraws the six PDFs in
`figures/` from the recorded results. To avoid updating those files, use a
different output directory:

```sh
python3 -B experiments/plot_figures.py --output-dir /tmp/csi_ode_figures_new
```

## Run experiments

See [the full reproduction instructions](experiments/README.md) for dependencies,
formal configurations, seed rules, metrics, and commands for neural experiments
and the endpoint-CF generation, fitting, evaluation, and audit stages.

Neural reruns require a fresh output directory outside this repository. The
endpoint-CF commands are run from `experiments/code/cf_endpoint/` and use a fresh
`outputs/` subdirectory there. Generated outputs and Python caches are ignored
by Git; the frozen paper result files must not be overwritten.

These experiments are synthetic mechanism diagnostics. Their parameterizations
and observation settings are described in the reproduction instructions; they
do not constitute a numerical proof of the statistical theorems. The archived
results are preserved, and a bit-identical full training rerun is not claimed.
