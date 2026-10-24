"""Train the daily VAE on training years and evaluate it on 2024.

Trains the small seeded MLP VAE (CPU) on the 2020-2023 included days only,
saves the models under ``models/`` (gitignored), samples validation-year
ensembles bound to 2024 reference dates and writes
``reports/vae_validation.json`` using the same masked metrics as the
baselines. Both beta variants are trained under identical conditions
(seed, epochs, architecture): beta = 1 (the original, posterior-collapsed
variant) and beta = 0.1 (KL down-weighted). The test year 2025 is not
touched.
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
BETAS = (1.0, 0.1)


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

    training = DayCollection.from_dataset(train)
    variants = {}
    for beta in BETAS:
        vae = DailyVAE(latent_dim=args.latent_dim, seed=SEED, beta=beta).fit(
            train, epochs=args.epochs
        )
        model_path = root / "models" / f"vae_q{args.latent_dim}_seed{SEED}_beta{beta}.pt"
        vae.save(model_path)
        variants[beta] = {
            "fit": {
                "years": "2020-2023 (training only)",
                "latent_dim": args.latent_dim,
                "seed": SEED,
                "beta": beta,
                "epochs": args.epochs,
                "final_train_mse": vae.history[-1]["mse"],
                "final_train_kl": vae.history[-1]["kl"],
                "train_reconstruction_mse": vae.reconstruction_mse(train),
                "validation_reconstruction_mse": vae.reconstruction_mse(validation),
            },
            "generated": summarize_collection(
                DayCollection.from_generated(vae.sample(len(validation.dates), validation)),
                training,
            ),
            "model_path": str(model_path),
        }

    report = {
        "representation": "k_calibrated 144-slot daily vectors; geometry from 2024 reference dates",
        "note": "beta = 1 is the original (posterior-collapsed) variant; beta = 0.1 down-weights the KL; identical seeds, epochs and architecture",
        **{f"vae_beta_{beta}": variant for beta, variant in variants.items()},
    }
    report_path = root / "reports" / "vae_validation.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2))

    baselines_path = root / "reports" / "baselines_validation.json"
    reference_rows = {}
    if baselines_path.is_file():
        baselines = json.loads(baselines_path.read_text())
        for name in ("observed_2024", "bootstrap", "ppca_q8"):
            reference_rows[name] = baselines[name]

    for beta, variant in variants.items():
        fit = variant["fit"]
        print(f"VAE q={args.latent_dim} beta={beta} ({args.epochs} epochs): "
              f"train mse {fit['final_train_mse']:.5f}, kl {fit['final_train_kl']:.2f}, "
              f"val recon mse {fit['validation_reconstruction_mse']:.5f}")
    rows = list(reference_rows.items()) + [
        (f"vae beta={beta}", variant["generated"]) for beta, variant in variants.items()
    ]
    for name, summary in rows:
        k = summary["k"]
        physical = summary["physical"]
        energy = summary["daily_energy_kwh_m2"]
        ramps = summary["abs_ramp_wm2_per_slot"]
        nn = summary["nearest_training_rms_k"]
        autocorr = summary["autocorrelation_k"]
        lag_one = autocorr[1] if 1 in autocorr else autocorr["1"]
        print(f"{name:16s}: k med {k['median']:.3f} | k>1 {physical['fraction_k_gt_1']:.1%} | "
              f"k<0 {physical['fraction_k_negative']:.1%} | energy med {energy['median']:.2f} kWh/m2 | "
              f"|dG| p95 {ramps['p95']:.0f} p99 {ramps['p99']:.0f} W/m2 | "
              f"lag-1 autocorr {lag_one:.3f} | "
              f"NN-RMS med {nn['median']:.3f}")
    print(f"Models: {[variant['model_path'] for variant in variants.values()]}; report: {report_path}")


if __name__ == "__main__":
    main()
