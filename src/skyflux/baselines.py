"""Statistical baselines on the daily cloud-factor representation.

Model input contract: one sample is one UTC day, a 144-slot vector of the
calibrated cloud factor ``k_calibrated``. ``k`` is finite exactly where the
slot carries a valid daylight measurement (status 1) and NaN elsewhere
(night and missing); a parallel boolean mask records that support. Nothing
is imputed — night is structurally absent from the model input and must
never appear as ``k = 0``, which would conflate geometry with overcast
physics.

Generators model cloudiness only. A generated 144-vector becomes physical
exclusively through a reference date from the comparison period: the
vector is read on that date's valid daylight slots and irradiance is
``G = k * G_clear_calibrated`` (zero elsewhere, per the physical
reference). Geometry is therefore assigned, never learned.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .clearsky import SPLIT_YEARS, STATUS_DAYLIGHT_VALID

# Day-to-split ids used in the daily npz; matches the enumeration order of
# clearsky.SPLIT_YEARS (train, validation, test) used by prepare_daily.py.
SPLIT_IDS = {"train": 0, "validation": 1, "test": 2}


@dataclass(frozen=True)
class DailyDataset:
    """Daily k profiles with validity masks, geometry and split assignment.

    ``k`` is (n_days, 144) float32, finite exactly where ``valid`` is true.
    ``g_clear_cal`` carries the calibrated clear-sky reference for every
    slot (zero at night) so irradiance can be reconstructed from any k
    profile combined with any reference date.
    """

    dates: np.ndarray  # (n_days,) datetime64[D]
    k: np.ndarray  # (n_days, 144) float32, NaN where not valid
    valid: np.ndarray  # (n_days, 144) bool, true on valid daylight slots
    g_clear_cal: np.ndarray  # (n_days, 144) float32
    include_day: np.ndarray  # (n_days,) bool
    split_id: np.ndarray  # (n_days,) int8

    def subset(self, split: str) -> "DailyDataset":
        """Included days of one chronological split (train/validation/test)."""
        try:
            wanted = SPLIT_IDS[split]
        except KeyError:
            raise ValueError(f"unknown split {split!r}; expected one of {sorted(SPLIT_IDS)}") from None
        selected = (self.split_id == wanted) & self.include_day
        return DailyDataset(
            dates=self.dates[selected],
            k=self.k[selected],
            valid=self.valid[selected],
            g_clear_cal=self.g_clear_cal[selected],
            include_day=self.include_day[selected],
            split_id=self.split_id[selected],
        )


@dataclass(frozen=True)
class GeneratedDay:
    """A generated cloud-factor profile bound to a reference date.

    ``k`` is the raw (144,) model output; ``daylight`` and ``g_clear_cal``
    come from the reference date. Physical irradiance is available only
    through ``irradiance()``, which reads ``k`` on the reference date's
    valid daylight slots — generated values on other slots are never used,
    so a model's night-slot output cannot leak into physics.
    """

    date: np.datetime64
    k: np.ndarray  # (144,) float32
    daylight: np.ndarray  # (144,) bool
    g_clear_cal: np.ndarray  # (144,) float32

    def irradiance(self) -> np.ndarray:
        """W/m^2 per slot: k on valid daylight slots, zero elsewhere."""
        return np.where(self.daylight, self.k * self.g_clear_cal, 0.0).astype(np.float32)


def load_daily_dataset(npz_path) -> DailyDataset:
    """Load the daily npz written by ``prepare_daily.py``.

    Validates the representation contract (k finite exactly on valid
    daylight slots) and the split-id/year correspondence, so an encoding
    change upstream cannot silently propagate into the generators.
    """
    npz = np.load(npz_path)
    k = np.asarray(npz["k_calibrated"], dtype=np.float32)
    valid = np.asarray(npz["status"]) == STATUS_DAYLIGHT_VALID
    if not np.array_equal(np.isfinite(k), valid):
        raise ValueError("k_calibrated must be finite exactly on valid daylight slots")
    dates = np.asarray(npz["dates"])
    split_id = np.asarray(npz["split_id"])
    years = dates.astype("datetime64[Y]").astype(int) + 1970
    for name, identifier in SPLIT_IDS.items():
        first, last = SPLIT_YEARS[name]
        in_split = years[split_id == identifier]
        if in_split.size and (in_split.min() < first or in_split.max() > last):
            raise ValueError(f"split_id {identifier} does not match years {first}-{last} for {name!r}")
    return DailyDataset(
        dates=dates,
        k=np.where(valid, k, np.nan).astype(np.float32),
        valid=valid,
        g_clear_cal=np.asarray(npz["g_clear_calibrated"], dtype=np.float32),
        include_day=np.asarray(npz["include_day"], dtype=bool),
        split_id=split_id,
    )


class BootstrapSampler:
    """Non-novel resampling baseline over whole training days.

    Draws days with replacement; each draw carries the day's own k profile,
    date and clear-sky reference, so decoded irradiance is exact by
    construction. This is the memorization floor for distributional
    metrics and the zero-novelty reference for diversity metrics: it
    reproduces training-day distributions, ramps and autocorrelation under
    resampling variability, but cannot generate new fluctuation patterns.
    Pass the training subset (``dataset.subset("train")``); nothing else
    should ever be exposed to a generator.
    """

    def __init__(self, dataset: DailyDataset, seed: int = 0):
        self._dataset = dataset
        self._rng = np.random.default_rng(seed)

    def sample(self, n_days: int) -> list[GeneratedDay]:
        """Draw ``n_days`` training days with replacement."""
        if len(self._dataset.dates) == 0:
            raise ValueError("cannot bootstrap-sample an empty dataset")
        rows = self._rng.integers(0, len(self._dataset.dates), size=n_days)
        return [
            GeneratedDay(
                date=self._dataset.dates[row],
                k=self._dataset.k[row].copy(),
                daylight=self._dataset.valid[row],
                g_clear_cal=self._dataset.g_clear_cal[row],
            )
            for row in rows
        ]


class PPCA:
    """Probabilistic PCA over masked daily k profiles, fitted by EM.

    Model: ``x = W z + mu + eps`` with ``z ~ N(0, I_q)`` (q latent
    components) and isotropic ``eps ~ N(0, sigma^2 I)``. Because the
    daylight support varies with season, each day enters the likelihood
    only through its valid slots ("missing by design"). With ``W_obs`` the
    rows of ``W`` on a day's observed slots and
    ``M = W_obs^T W_obs + sigma^2 I``:

        E-step:  E[z] = M^-1 W_obs^T (x_obs - mu_obs)
                 E[z z^T] = sigma^2 M^-1 + E[z] E[z]^T
        M-step:  W_j = (sum_{i observes j} Delta_ij E[z_i]^T)
                      (sum_{i observes j} E[z_i z_i^T])^-1
                 sigma^2 = (1 / n_observed) sum_i [ ||Delta_i - W_new E[z_i]||^2
                           + sigma^2 tr(W_new,obs^T W_new,obs M^-1) ]

    A day never touches the rows of ``W`` on its night slots, so no latent
    capacity is spent on day length. ``mu`` is the masked per-slot training
    mean and stays fixed (standard PPCA centering). Days are grouped by
    identical validity mask, which vectorizes the EM per pattern instead of
    per day. Convergence is monitored on the observed-data log-likelihood.

    Generation draws ``z`` from the prior — genuinely generative, not a
    reconstruction of training days — and returns raw 144-slot profiles
    that become physical only against a reference date's daylight slots
    (see ``GeneratedDay``). ``include_noise=True`` adds the isotropic
    residual ``sigma * eps`` so its artifacts can be compared against the
    noiseless variant rather than hidden inside the model. Values are never
    clipped: negative or extreme k are model output and get reported.
    """

    def __init__(
        self,
        n_components: int = 8,
        seed: int = 0,
        max_iter: int = 500,
        tol: float = 1e-9,
    ):
        self.n_components = int(n_components)
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self._rng = np.random.default_rng(seed)
        self.mu: np.ndarray | None = None  # (144,) fixed masked mean
        self.W: np.ndarray | None = None  # (144, q)
        self.sigma2: float | None = None
        self.log_likelihood: float | None = None
        self.n_iterations: int = 0

    # -- fitting -----------------------------------------------------------

    def fit(self, dataset: DailyDataset) -> "PPCA":
        """Fit on the dataset's days; only valid daylight slots are used."""
        if len(dataset.dates) == 0:
            raise ValueError("cannot fit PPCA on an empty dataset")
        q = self.n_components
        valid = dataset.valid
        # Slots valid on no training day (deep-night slots are never valid
        # at this latitude) are outside the model: mu = 0 and W = 0 there,
        # so generated profiles read 0 — any physical reference date masks
        # those slots out anyway, because the daylight geometry is a
        # deterministic function of the date.
        observed_slot = valid.any(axis=0)
        if not observed_slot.any():
            raise ValueError("no valid slots in the training dataset")
        mu = np.zeros(144, dtype=np.float64)
        mu[observed_slot] = np.nanmean(
            np.where(valid[:, observed_slot], dataset.k[:, observed_slot], np.nan), axis=0
        )
        delta_all = np.where(valid, dataset.k - mu, 0.0).astype(np.float64)

        # Group days by identical validity mask; every EM matrix is shared
        # within a group.
        patterns, inverse = np.unique(valid, axis=0, return_inverse=True)
        groups = [np.flatnonzero(inverse == p) for p in range(len(patterns))]
        identity_q = np.eye(q)

        scale = max(float(np.nanmean(np.where(valid, (dataset.k - mu) ** 2, np.nan))), 1e-6)
        W = 0.1 * np.sqrt(scale) * self._rng.standard_normal((144, q))
        W[~observed_slot] = 0.0
        sigma2 = 0.1 * scale

        def observed_log_likelihood() -> float:
            """Sum of per-day Gaussian log-likelihoods on observed slots."""
            total = 0.0
            for pattern, rows in zip(patterns, groups):
                obs = np.flatnonzero(pattern)
                n_p = len(rows)
                W_obs = W[obs, :]
                covariance = W_obs @ W_obs.T + sigma2 * np.eye(len(obs))
                sign, logdet = np.linalg.slogdet(covariance)
                solved = np.linalg.solve(covariance, delta_all[np.ix_(rows, obs)].T)
                # Quadratic form sum_i d_i^T C^-1 d_i, NOT the squared norm
                # of C^-1 d (which would be d^T C^-2 d).
                quad = float(np.sum(solved * delta_all[np.ix_(rows, obs)].T))
                total -= 0.5 * (
                    n_p * len(obs) * np.log(2.0 * np.pi) + n_p * logdet + quad
                )
            return total

        previous_ll = observed_log_likelihood()
        iterations = 0
        for iterations in range(1, self.max_iter + 1):
            # E-step (per pattern) and the W sufficient statistics.
            cross = np.zeros((144, q))  # sum_i Delta_ij E[z_i]^T, per slot j
            second = np.zeros((144, q, q))  # sum_{i observes j} E[z_i z_i^T]
            posteriors = []  # (rows, obs, E[z], M^-1) kept for the sigma^2 step
            for pattern, rows in zip(patterns, groups):
                obs = np.flatnonzero(pattern)
                W_obs = W[obs, :]
                delta = delta_all[np.ix_(rows, obs)]
                M = W_obs.T @ W_obs + sigma2 * identity_q
                M_inv = np.linalg.inv(M)
                ez = np.linalg.solve(M, W_obs.T @ delta.T).T  # (n_p, q)
                cross[obs] += delta.T @ ez
                second[obs] += sigma2 * M_inv * len(rows) + ez.T @ ez
                posteriors.append((obs, ez, M_inv, len(rows)))

            # M-step for W: solve each slot's small q x q system. Slots
            # observed on no day keep all-zero rows (their system is
            # singular by construction and carries no information).
            W = np.zeros((144, q))
            W[observed_slot] = np.linalg.solve(
                second[observed_slot], cross[observed_slot, :, None]
            )[:, :, 0]

            # M-step for sigma^2: new W against the old posterior moments.
            noise_num = 0.0
            for rows, (obs, ez, M_inv, _n_p) in zip(groups, posteriors):
                W_obs = W[obs, :]
                delta = delta_all[np.ix_(rows, obs)]
                noise_num += float(np.sum((delta - ez @ W_obs.T) ** 2))
                noise_num += sigma2 * len(rows) * float(np.sum(M_inv * (W_obs.T @ W_obs)))
            sigma2 = noise_num / int(valid.sum())

            current_ll = observed_log_likelihood()
            if abs(current_ll - previous_ll) <= self.tol * (abs(previous_ll) + 1.0):
                previous_ll = current_ll
                break
            previous_ll = current_ll

        self.mu = mu
        self.W = W
        self.sigma2 = float(sigma2)
        self.log_likelihood = float(previous_ll)
        self.n_iterations = iterations
        return self

    # -- diagnostics -------------------------------------------------------

    def latent_variances(self) -> np.ndarray:
        """Variances captured along the latent axes, largest first.

        These are the eigenvalues of ``W^T W + sigma^2 I`` — the implied
        covariance of the latent space under the fitted model.
        """
        if self.W is None:
            raise ValueError("PPCA must be fitted first")
        return np.linalg.eigvalsh(self.W.T @ self.W + self.sigma2 * np.eye(self.n_components))[::-1]

    # -- generation --------------------------------------------------------

    def sample(
        self,
        n_days: int,
        reference: DailyDataset,
        include_noise: bool = False,
    ) -> list[GeneratedDay]:
        """Draw ``n_days`` prior samples bound to random reference dates.

        Reference dates are drawn uniformly from ``reference`` (pass the
        comparison-period subset); they supply the daylight mask and the
        clear-sky reference used to decode irradiance, keeping geometry
        assigned rather than modelled. The internal random generator
        advances across calls; identical call sequences with the same seed
        reproduce identical samples.
        """
        if self.W is None or self.mu is None:
            raise ValueError("PPCA must be fitted before sampling")
        if len(reference.dates) == 0:
            raise ValueError("reference dataset is empty")
        z = self._rng.standard_normal((n_days, self.n_components))
        k = z @ self.W.T + self.mu
        if include_noise:
            k = k + self._rng.normal(0.0, np.sqrt(self.sigma2), size=k.shape)
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
