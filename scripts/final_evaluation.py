"""Final held-out evaluation on the 2025 test year.

Model choices are frozen from the 2024 validation phase: the bootstrap
memorization floor, PPCA q=8 with and without its isotropic residual, and
the variance-matched VAE. Every generator was fitted on 2020-2023 only;
nothing about 2025 informed any model or configuration choice, and this
script performs no tuning — it fits the frozen configurations
deterministically from fixed seeds, samples ensembles bound to 2025
reference dates and compares them with the masked metrics.

Writes ``reports/final_evaluation_2025.json`` and
``figures/final_evaluation_2025.png``. Development variants (VAE beta=1,
beta=0.1) remain documented in ``reports/vae_validation.json`` and are
not part of the frozen comparison.
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
    ramp_magnitudes,
    summarize_collection,
)
from skyflux.vae import EPOCHS, DailyVAE

SEED = 0
N_COMPONENTS = 8


def collections_for_test(train, test, root: Path) -> dict:
    """Frozen generators, each sampled as a test-year-sized ensemble."""
    n_days = len(test.dates)

    ppca = PPCA(n_components=N_COMPONENTS, seed=SEED).fit(train)

    # Residual variance of k around its masked mean, over observed slots
    # only (midnight slots are never valid and would warn on nanmean).
    observed_slot = train.valid.any(axis=0)
    masked_mean = np.full(144, np.nan)
    masked_mean[observed_slot] = np.nanmean(
        np.where(train.valid[:, observed_slot], train.k[:, observed_slot], np.nan), axis=0
    )
    sigma2_hat = float(
        np.nanmean(
            np.where(
                train.valid[:, observed_slot],
                (train.k - masked_mean)[:, observed_slot] ** 2,
                np.nan,
            )
        )
    )
    model_path = root / "models" / f"vae_q{N_COMPONENTS}_seed{SEED}_variance_matched.pt"
    if model_path.is_file():
        vae = DailyVAE.load(model_path, seed=SEED)
    else:
        # Deterministic refit identical to the saved model (same seed).
        vae = DailyVAE(latent_dim=N_COMPONENTS, seed=SEED, sigma2=sigma2_hat).fit(
            train, epochs=EPOCHS
        )

    ensembles = {
        "bootstrap": BootstrapSampler(train, seed=SEED).sample(n_days),
        "ppca_q8": ppca.sample(n_days, test),
        "ppca_q8_with_noise": ppca.sample(n_days, test, include_noise=True),
        "vae_variance_matched": vae.sample(n_days, test),
    }
    generated = {name: DayCollection.from_generated(days) for name, days in ensembles.items()}
    generated["observed_2025"] = DayCollection.from_dataset(test)
    return generated


def save_figure(target: Path, collections: dict, labels: dict) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(13.5, 7.5))

    ax = axes[0, 0]
    bins = np.linspace(0.0, 2.5, 51)
    for name, (color, label) in labels.items():
        collection = collections[name]
        values = collection.k[collection.valid]
        ax.hist(values, bins=bins, density=True, histtype="step", linewidth=1.5,
                label=label, color=color)
    ax.set_yscale("log")
    ax.set_xlabel("k (calibrated, valid daylight slots)")
    ax.set_ylabel("density (log)")
    ax.set_title("Cloud-factor distributions (2025)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[0, 1]
    for name, (color, label) in labels.items():
        energy = np.sort(daily_energy(collections[name]) / 3.6e6)
        ax.step(energy, np.arange(1, len(energy) + 1) / len(energy), label=label, color=color)
    ax.set_xlabel("daily irradiation (kWh/m$^2$)")
    ax.set_ylabel("empirical CDF")
    ax.set_title("Daily integrated irradiance")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[1, 0]
    bins = np.linspace(0.0, 400.0, 41)
    for name, (color, label) in labels.items():
        collection = collections[name]
        values = np.abs(np.diff(collection.ghi(), axis=1))[
            collection.valid[:, :-1] & collection.valid[:, 1:]
        ]
        ax.hist(values, bins=bins, density=True, histtype="step", linewidth=1.5,
                label=label, color=color)
    ax.set_yscale("log")
    ax.set_xlabel("|dG| per 10 min (W/m$^2$)")
    ax.set_ylabel("density (log)")
    ax.set_title("Ramp magnitudes")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[1, 1]
    lags = np.arange(1, 7)
    width = 0.15
    for offset, (name, (color, label)) in enumerate(labels.items()):
        values = [autocorrelation_k(collections[name], max_lag=6)[lag] for lag in lags]
        ax.bar(lags + (offset - 2) * width, values, width=width, label=label, color=color)
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

    npz_path = (
        root / "data/processed"
        / f"skyflux_daily_{args.station}_{FIRST_YEAR}_{LAST_YEAR}.npz"
    )
    dataset = load_daily_dataset(npz_path)
    train = dataset.subset("train")
    test = dataset.subset("test")

    collections = collections_for_test(train, test, root)
    training_pool = DayCollection.from_dataset(train)
    labels = {
        "observed_2025": ("tab:blue", "observed 2025"),
        "bootstrap": ("tab:orange", "bootstrap (train days)"),
        "ppca_q8": ("tab:green", "PPCA q=8"),
        "ppca_q8_with_noise": ("tab:red", "PPCA q=8 + residual"),
        "vae_variance_matched": ("tab:purple", "VAE variance-matched"),
    }

    report = {
        "protocol": (
            "held-out 2025 test year; all models fitted on 2020-2023 only; "
            "frozen configurations, fixed seeds, no tuning on test data; "
            "ensembles sampled against 2025 reference dates"
        ),
        "n_test_days": int(len(test.dates)),
        "collections": {
            name: summarize_collection(collection, training_pool)
            for name, collection in collections.items()
        },
    }
    report_path = root / "reports" / "final_evaluation_2025.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2))
    save_figure(root / "figures" / "final_evaluation_2025.png", collections, labels)

    for name, (color, label) in labels.items():
        summary = report["collections"][name]
        k = summary["k"]
        physical = summary["physical"]
        energy = summary["daily_energy_kwh_m2"]
        ramps = summary["abs_ramp_wm2_per_slot"]
        nn = summary["nearest_training_rms_k"]
        print(f"{name:20s}: k med {k['median']:.3f} | k>1 {physical['fraction_k_gt_1']:.1%} | "
              f"k<0 {physical['fraction_k_negative']:.1%} | energy med {energy['median']:.2f} kWh/m2 | "
              f"|dG| p95 {ramps['p95']:.0f} p99 {ramps['p99']:.0f} W/m2 | "
              f"lag-1 {summary['autocorrelation_k'][1]:.3f} | "
              f"NN-RMS med {nn['median']:.3f}")
    print(f"Outputs: {report_path}, {root / 'figures' / 'final_evaluation_2025.png'}")


if __name__ == "__main__":
    main()
