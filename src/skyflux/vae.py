"""Small MLP variational autoencoder for daily cloud-factor profiles.

Same representation and decoding contract as the statistical baselines:
the model sees 144-slot ``k_calibrated`` days with invalid slots zeroed at
the input, the reconstruction loss is computed only on valid daylight
slots, and generated 144-vectors become physical only through a reference
date's daylight mask (``GeneratedDay``). Values are never clipped — the
decoder may emit negative or extreme k and the evaluation reports them.

Generative model: encoder ``q(z|x) = N(mu(x), diag(exp(logvar(x))))``,
standard-normal prior, Gaussian decoder with fixed unit variance. The
training objective is a beta-weighted, per-day-normalized variational
bound

    L = E_q[ ||x_obs - f(z)_obs||^2 / m ] + beta * KL(q(z|x) || N(0, I)),

not the plain evidence lower bound: the per-day normalization (m = the
day's valid slot count) reweights the data so each day, not each slot,
contributes equally (days are the generation unit), and beta rescales
the KL. Matching term by term against the exact Gaussian ELBO shows the
implied observation variance is day-dependent: per unit of summed
squared error the exact ELBO weights the KL by 2 sigma^2, while this
objective weights it by beta * m, so the implied variance is
sigma^2 = beta * m / 2. Equal day weighting and a single exact-ELBO
scale are therefore incompatible: the variance-matched choice is a
per-day weight beta_i = 2 * sigma_hat^2 / m_i with sigma_hat^2 the
empirical residual variance of k (~0.15 here; pass ``sigma2`` to enable
this mode). A constant beta = 1 implies sigma^2 of 20-45 W/m^2-scale —
far above the residual, which is why that variant posterior-collapses.
Training is CPU-seeded and deterministic: fixed torch seed at
construction, a seeded numpy generator for minibatch shuffling and
sampling.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .baselines import DailyDataset, GeneratedDay

LATENT_DIM = 8
HIDDEN_DIMS = (256, 128)
EPOCHS = 200
BATCH_SIZE = 64
LEARNING_RATE = 1e-3


class _MLPVAE(nn.Module):
    """Encoder/decoder networks; input day vectors have invalid slots zeroed."""

    def __init__(self, latent_dim: int, hidden: tuple[int, ...]):
        super().__init__()
        first, second = hidden
        self.encoder = nn.Sequential(
            nn.Linear(144, first),
            nn.GELU(),
            nn.Linear(first, second),
            nn.GELU(),
        )
        self.to_mean = nn.Linear(second, latent_dim)
        self.to_logvar = nn.Linear(second, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, second),
            nn.GELU(),
            nn.Linear(second, first),
            nn.GELU(),
            nn.Linear(first, 144),
        )

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.encoder(x)
        return self.to_mean(hidden), self.to_logvar(hidden)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def loss_terms(self, x: torch.Tensor, mask: torch.Tensor, beta=1.0):
        """Masked per-day MSE and beta-weighted KL, averaged over days.

        ``beta`` is either a scalar or a per-day tensor (the
        variance-matched mode uses ``2 * sigma^2 / m_i`` per day).
        Invalid slots are zeroed on the input here so the encoder can
        never see them, regardless of what the caller passes; the mask
        additionally zeroes the reconstruction loss on those slots.
        """
        x = x * mask
        mean, logvar = self.encode(x)
        std = torch.exp(0.5 * logvar)
        z = mean + std * torch.randn_like(std)
        reconstruction = self.decode(z)
        slot_count = mask.sum(dim=1).clamp(min=1.0)
        per_day_mse = (((reconstruction - x) ** 2) * mask).sum(dim=1) / slot_count
        mse = per_day_mse.mean()
        kl_per_day = (-0.5 * (1.0 + logvar - mean**2 - torch.exp(logvar))).sum(dim=1)
        kl = kl_per_day.mean()
        if torch.is_tensor(beta):
            total = mse + (beta * kl_per_day).mean()
        else:
            total = mse + beta * kl
        return total, mse, kl


class DailyVAE:
    """Trains the VAE on training-year days and samples new days.

    ``fit`` consumes the training subset only (as with the baselines,
    nothing else should ever be passed). ``sample`` draws ``z`` from the
    standard-normal prior, decodes 144-slot profiles and binds them to
    random reference dates from the comparison-period subset — the same
    geometry-assigned decoding contract as ``BootstrapSampler``/``PPCA``.
    """

    def __init__(
        self,
        latent_dim: int = LATENT_DIM,
        hidden: tuple[int, ...] = HIDDEN_DIMS,
        seed: int = 0,
        beta: float = 1.0,
        sigma2: float | None = None,
    ):
        self.latent_dim = int(latent_dim)
        self.hidden = tuple(hidden)
        self.seed = int(seed)
        self.beta = float(beta)
        # Variance-matched mode: per-day KL weights 2*sigma2/m_i (exact
        # ELBO scale per day, equal day weighting preserved).
        self.sigma2 = None if sigma2 is None else float(sigma2)
        torch.manual_seed(self.seed)
        self._model = _MLPVAE(self.latent_dim, self.hidden)
        self._rng = np.random.default_rng(self.seed)
        self.history: list[dict] = []

    def fit(
        self,
        dataset: DailyDataset,
        epochs: int = EPOCHS,
        batch_size: int = BATCH_SIZE,
        learning_rate: float = LEARNING_RATE,
    ) -> "DailyVAE":
        if len(dataset.dates) == 0:
            raise ValueError("cannot train the VAE on an empty dataset")
        x = torch.tensor(np.where(dataset.valid, dataset.k, 0.0), dtype=torch.float32)
        mask = torch.tensor(dataset.valid, dtype=torch.float32)
        optimizer = torch.optim.Adam(self._model.parameters(), lr=learning_rate)
        self.history = []
        n_days = len(x)
        for epoch in range(1, int(epochs) + 1):
            order = self._rng.permutation(n_days)
            epoch_mse = epoch_kl = 0.0
            batches = 0
            for start in range(0, n_days, int(batch_size)):
                rows = torch.tensor(order[start : start + int(batch_size)])
                optimizer.zero_grad()
                if self.sigma2 is not None:
                    beta = 2.0 * self.sigma2 / mask[rows].sum(dim=1).clamp(min=1.0)
                else:
                    beta = self.beta
                loss, mse, kl = self._model.loss_terms(x[rows], mask[rows], beta=beta)
                loss.backward()
                optimizer.step()
                epoch_mse += mse.item()
                epoch_kl += kl.item()
                batches += 1
            self.history.append(
                {
                    "epoch": epoch,
                    "mse": epoch_mse / batches,
                    "kl": epoch_kl / batches,
                }
            )
        return self

    def reconstruction_mse(self, dataset: DailyDataset) -> float:
        """Masked per-day MSE of the posterior-mean reconstruction (no sampling)."""
        with torch.no_grad():
            x = torch.tensor(np.where(dataset.valid, dataset.k, 0.0), dtype=torch.float32)
            mask = torch.tensor(dataset.valid, dtype=torch.float32)
            mean, _logvar = self._model.encode(x * mask)
            reconstruction = self._model.decode(mean)
            slot_count = mask.sum(dim=1).clamp(min=1.0)
            per_day = (((reconstruction - x) ** 2) * mask).sum(dim=1) / slot_count
            return float(per_day.mean())

    def sample(
        self, n_days: int, reference: DailyDataset, include_noise: bool = False
    ) -> list[GeneratedDay]:
        """Draw ``n_days`` prior samples bound to random reference dates.

        ``include_noise=True`` adds unit-variance Gaussian observation
        noise per slot (the decoder's assumed output sigma of 1), mirroring
        the PPCA residual-noise variant for artifact comparison; like the
        baselines, nothing is clipped.
        """
        if len(reference.dates) == 0:
            raise ValueError("reference dataset is empty")
        z = torch.tensor(
            self._rng.standard_normal((n_days, self.latent_dim)), dtype=torch.float32
        )
        with torch.no_grad():
            k = self._model.decode(z).numpy()
        if include_noise:
            k = k + self._rng.normal(0.0, 1.0, size=k.shape)
        rows = self._rng.integers(0, len(reference.dates), size=n_days)
        return [
            GeneratedDay(
                date=reference.dates[row],
                k=k[i].astype(np.float32),
                daylight=reference.valid[row],
                g_clear_cal=reference.g_clear_cal[row],
            )
            for i, row in enumerate(rows)
        ]

    def save(self, path) -> None:
        """Persist weights and configuration (model artifacts stay out of Git)."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": self._model.state_dict(),
                "config": {
                    "latent_dim": self.latent_dim,
                    "hidden": list(self.hidden),
                    "seed": self.seed,
                    "beta": self.beta,
                    "sigma2": self.sigma2,
                },
            },
            path,
        )

    @classmethod
    def load(cls, path, seed: int = 0) -> "DailyVAE":
        payload = torch.load(path, weights_only=True)
        config = payload["config"]
        instance = cls(
            latent_dim=config["latent_dim"],
            hidden=tuple(config["hidden"]),
            seed=seed,
            beta=config.get("beta", 1.0),
            sigma2=config.get("sigma2"),
        )
        instance._model.load_state_dict(payload["state_dict"])
        return instance
