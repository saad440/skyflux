"""Fit the statistical baselines and evaluate them on the validation year.

Loads the daily npz written by ``prepare_daily.py``, fits PPCA (masked-data
EM) on the training years 2020-2023 only, samples validation-year-sized
ensembles bound to 2024 reference dates, and compares observed 2024 days,
bootstrap resamples and PPCA samples with the masked evaluation metrics.
Writes ``reports/baselines_validation.json`` and
``figures/baselines_validation.png``. The test year 2025 is not touched.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from skyflux.baselines import BootstrapSampler, PPCA, load_daily_dataset
from skyflux.dwd import FIRST_YEAR, LAST_YEAR
from skyflux.evaluation import (
    DayCollection,
    autocorrelation_k,
    daily_energy,
    summarize_collection,
)

N_COMPONENTS = 8
SEED = 0
PPCA_SENSITIVITY_Q = (4, 12)


def save_figure(target: Path, real, bootstrap, ppca) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    series = (
        (real, "observed 2024", "tab:blue"),
        (bootstrap, "bootstrap (train days)", "tab:orange"),
        (ppca, f"PPCA q={N_COMPONENTS}", "tab:green"),
    )
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 7.5))

    ax = axes[0, 0]
    bins = np.linspace(0.0, 2.5, 51)
    for collection, label, color in series:
        values = collection.k[collection.valid]
        ax.hist(values, bins=bins, density=True, histtype="step", linewidth=1.5,
                label=label, color=color)
    ax.set_yscale("log")
    ax.set_xlabel("k (calibrated, valid daylight slots)")
    ax.set_ylabel("density (log)")
    ax.set_title("Cloud-factor distributions")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[0, 1]
    for collection, label, color in series:
        energy = np.sort(daily_energy(collection) / 3.6e6)
        ax.step(energy, np.arange(1, len(energy) + 1) / len(energy), label=label, color=color)
    ax.set_xlabel("daily irradiation (kWh/m$^2$)")
    ax.set_ylabel("empirical CDF")
    ax.set_title("Daily integrated irradiance")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[1, 0]
    bins = np.linspace(0.0, 400.0, 41)
    for collection, label, color in series:
        values = np.abs(
            np.diff(collection.ghi(), axis=1)
        )[collection.valid[:, :-1] & collection.valid[:, 1:]]
        ax.hist(values, bins=bins, density=True, histtype="step", linewidth=1.5,
                label=label, color=color)
    ax.set_yscale("log")
    ax.set_xlabel("|dG| per 10 min (W/m$^2$, consecutive valid slots)")
    ax.set_ylabel("density (log)")
    ax.set_title("Ramp magnitudes")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[1, 1]
    lags = np.arange(1, 7)
    width = 0.25
    for offset, (collection, label, color) in enumerate(series):
        values = [autocorrelation_k(collection, max_lag=6)[lag] for lag in lags]
        ax.bar(lags + (offset - 1) * width, values, width=width, label=label, color=color)
    ax.set_xlabel("lag (10-minute slots)")
    ax.set_ylabel("pooled autocorrelation of k")
    ax.set_title("Short-lag temporal structure")
    ax.set_xticks(lags)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    fig.tight_layout()
    target.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(target, dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--station", default="00342", help="Five-digit DWD station ID")
    parser.add_argument("--root", type=Path, default=Path.cwd(), help="Project root")
    args = parser.parse_args()
    root: Path = args.root

    npz_path = root / "data/processed" / f"skyflux_daily_{args.station}_{FIRST_YEAR}_{LAST_YEAR}.npz"
    dataset = load_daily_dataset(npz_path)
    train = dataset.subset("train")
    validation = dataset.subset("validation")
    n_days = len(validation.dates)

    ppca = PPCA(n_components=N_COMPONENTS, seed=SEED).fit(train)
    sensitivity = {}
    for q in PPCA_SENSITIVITY_Q:
        model = PPCA(n_components=q, seed=SEED).fit(train)
        sensitivity[f"q={q}"] = {
            "log_likelihood": model.log_likelihood,
            "sigma2": model.sigma2,
            "n_iterations": model.n_iterations,
            "latent_variances": [float(v) for v in model.latent_variances()],
        }

    bootstrap_days = BootstrapSampler(train, seed=SEED).sample(n_days)
    ppca_days = ppca.sample(n_days, validation)
    ppca_noisy_days = ppca.sample(n_days, validation, include_noise=True)

    real = DayCollection.from_dataset(validation)
    training = DayCollection.from_dataset(train)
    bootstrap = DayCollection.from_generated(bootstrap_days)
    ppca_collection = DayCollection.from_generated(ppca_days)
    ppca_noisy = DayCollection.from_generated(ppca_noisy_days)

    report = {
        "representation": "k_calibrated 144-slot daily vectors; geometry from reference dates",
        "fit": {
            "years": "2020-2023 (training only)",
            "ppca": {
                "n_components": N_COMPONENTS,
                "seed": SEED,
                "log_likelihood": ppca.log_likelihood,
                "log_likelihood_per_valid_slot": ppca.log_likelihood / int(train.valid.sum()),
                "sigma2": ppca.sigma2,
                "n_iterations": ppca.n_iterations,
                "latent_variances": [float(v) for v in ppca.latent_variances()],
            },
            "ppca_sensitivity": sensitivity,
        },
        "sampling": {
            "n_days_per_ensemble": n_days,
            "seed": SEED,
            "ppca_reference_pool": "2024 validation dates (geometry assigned, not modelled)",
            "bootstrap_reference_pool": "sampled training dates (inherent to resampling)",
        },
        "observed_2024": summarize_collection(real, training),
        "bootstrap": summarize_collection(bootstrap, training),
        "ppca_q8": summarize_collection(ppca_collection, training),
        "ppca_q8_with_noise": summarize_collection(ppca_noisy, training),
    }
    report_path = root / "reports" / "baselines_validation.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2))

    save_figure(
        root / "figures" / "baselines_validation.png", real, bootstrap, ppca_collection
    )

    print(f"PPCA q={N_COMPONENTS}: ll/slot {report['fit']['ppca']['log_likelihood_per_valid_slot']:+.3f}, "
          f"sigma^2 {ppca.sigma2:.5f}, iters {ppca.n_iterations}")
    for name, collection in (
        ("observed 2024", real),
        ("bootstrap", bootstrap),
        ("ppca q=8", ppca_collection),
        ("ppca q=8 +noise", ppca_noisy),
    ):
        summary = summarize_collection(collection, training)
        k = summary["k"]
        physical = summary["physical"]
        energy = summary["daily_energy_kwh_m2"]
        ramps = summary["abs_ramp_wm2_per_slot"]
        nn = summary["nearest_training_rms_k"]
        print(f"{name:16s}: k med {k['median']:.3f} | k>1 {physical['fraction_k_gt_1']:.1%} | "
              f"k<0 {physical['fraction_k_negative']:.1%} | energy med {energy['median']:.2f} kWh/m2 | "
              f"|dG| p95 {ramps['p95']:.0f} p99 {ramps['p99']:.0f} W/m2 | "
              f"lag-1 autocorr {summary['autocorrelation_k'][1]:.3f} | "
              f"NN-RMS med {nn['median']:.3f}")
    print(f"Outputs: {report_path}, {root / 'figures' / 'baselines_validation.png'}")


if __name__ == "__main__":
    main()
