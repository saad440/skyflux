"""Train the daily VAE on training years and evaluate it on 2024.

Trains the small seeded MLP VAE (CPU) on the 2020-2023 included days only,
saves the model under ``models/`` (gitignored), samples a validation-year
ensemble bound to 2024 reference dates and writes
``reports/vae_validation.json`` using the same masked metrics as the
baselines. The test year 2025 is not touched.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from skyflux.baselines import load_daily_dataset
from skyflux.dwd import FIRST_YEAR, LAST_YEAR
from skyflux.evaluation import DayCollection, summarize_collection
from skyflux.vae import LATENT_DIM, EPOCHS, DailyVAE

SEED = 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--station", default="00342", help="Five-digit DWD station ID")
    parser.add_argument("--root", type=Path, default=Path.cwd(), help="Project root")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--latent-dim", type=int, default=LATENT_DIM)
    args = parser.parse_args()
    root: Path = args.root

    npz_path = (
        root / "data/processed"
        / f"skyflux_daily_{args.station}_{FIRST_YEAR}_{LAST_YEAR}.npz"
    )
    dataset = load_daily_dataset(npz_path)
    train = dataset.subset("train")
    validation = dataset.subset("validation")

    vae = DailyVAE(latent_dim=args.latent_dim, seed=SEED).fit(train, epochs=args.epochs)
    model_path = root / "models" / f"vae_q{args.latent_dim}_seed{SEED}.pt"
    vae.save(model_path)

    generated = DayCollection.from_generated(vae.sample(len(validation.dates), validation))
    training = DayCollection.from_dataset(train)
    report = {
        "representation": "k_calibrated 144-slot daily vectors; geometry from 2024 reference dates",
        "fit": {
            "years": "2020-2023 (training only)",
            "latent_dim": args.latent_dim,
            "seed": SEED,
            "epochs": args.epochs,
            "final_train_mse": vae.history[-1]["mse"],
            "final_train_kl": vae.history[-1]["kl"],
            "train_reconstruction_mse": vae.reconstruction_mse(train),
            "validation_reconstruction_mse": vae.reconstruction_mse(validation),
        },
        "vae": summarize_collection(generated, training),
        "model_path": str(model_path),
    }
    report_path = root / "reports" / "vae_validation.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2))

    summary = report["vae"]
    print(f"VAE q={args.latent_dim} ({args.epochs} epochs): "
          f"train mse {report['fit']['final_train_mse']:.5f}, kl {report['fit']['final_train_kl']:.2f}, "
          f"val recon mse {report['fit']['validation_reconstruction_mse']:.5f}")
    k = summary["k"]
    physical = summary["physical"]
    energy = summary["daily_energy_kwh_m2"]
    ramps = summary["abs_ramp_wm2_per_slot"]
    nn = summary["nearest_training_rms_k"]
    print(f"{'vae q8':16s}: k med {k['median']:.3f} | k>1 {physical['fraction_k_gt_1']:.1%} | "
          f"k<0 {physical['fraction_k_negative']:.1%} | energy med {energy['median']:.2f} kWh/m2 | "
          f"|dG| p95 {ramps['p95']:.0f} p99 {ramps['p99']:.0f} W/m2 | "
          f"lag-1 autocorr {summary['autocorrelation_k'][1]:.3f} | "
          f"NN-RMS med {nn['median']:.3f}")
    print(f"Model: {model_path}; report: {report_path}")


if __name__ == "__main__":
    main()
