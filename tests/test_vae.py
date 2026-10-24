"""Unit tests for the VAE on synthetic daily profiles (tiny network, CPU)."""
import numpy as np
import pytest
import torch

from skyflux.baselines import DailyDataset
from skyflux.vae import DailyVAE

from tests.test_baselines import make_dataset


def test_vae_training_is_deterministic():
    dataset = make_dataset(n_days=40)
    first = DailyVAE(latent_dim=2, hidden=(16, 8), seed=3).fit(dataset, epochs=3, batch_size=8)
    second = DailyVAE(latent_dim=2, hidden=(16, 8), seed=3).fit(dataset, epochs=3, batch_size=8)
    assert [entry["mse"] for entry in first.history] == [
        entry["mse"] for entry in second.history
    ]
    assert [entry["kl"] for entry in first.history] == [
        entry["kl"] for entry in second.history
    ]
    samples_a = first.sample(5, dataset)
    samples_b = second.sample(5, dataset)
    for a, b in zip(samples_a, samples_b):
        np.testing.assert_array_equal(a.k, b.k)
        assert a.date == b.date


def test_vae_variance_matched_per_day_weights():
    vae = DailyVAE(latent_dim=2, hidden=(16, 8), seed=0, sigma2=0.15)
    x = torch.rand(2, 144)
    mask = torch.zeros(2, 144)
    mask[0, :40] = 1.0
    mask[1, :90] = 1.0
    weights = 2.0 * 0.15 / mask.sum(dim=1)
    total, mse, kl = vae._model.loss_terms(x, mask, beta=weights)
    assert torch.isfinite(total)
    # Fewer valid slots -> cheaper KL for that day (weight 2 sigma^2 / m).
    assert weights[0] > weights[1]
    # The scalar path keeps its meaning.
    torch.manual_seed(0)
    total, mse, kl = vae._model.loss_terms(x, mask, beta=0.1)
    assert total.item() == pytest.approx(mse.item() + 0.1 * kl.item())


def test_vae_beta_weights_kl():
    vae = DailyVAE(latent_dim=2, hidden=(16, 8), seed=0, beta=0.0)
    x = torch.rand(2, 144)
    mask = torch.ones(2, 144)
    torch.manual_seed(1)
    total, mse, kl = vae._model.loss_terms(x, mask, beta=vae.beta)
    assert kl.item() > 0
    assert total.item() == pytest.approx(mse.item())  # beta = 0 drops the KL


def test_vae_loss_ignores_invalid_slots():
    vae = DailyVAE(latent_dim=2, hidden=(16, 8), seed=0)
    x = torch.zeros(2, 144)
    mask = torch.zeros(2, 144)
    mask[:, :10] = 1.0
    x[:, :10] = 0.5
    x[0, 50:] = 3.0  # garbage on invalid slots
    torch.manual_seed(1)  # identical reparameterization noise
    _, mse_a, _ = vae._model.loss_terms(x.clone(), mask)
    x[0, 50:] = -7.0  # different garbage, same valid content
    torch.manual_seed(1)
    _, mse_b, _ = vae._model.loss_terms(x, mask)
    assert mse_a == mse_b  # invalid slots never enter the loss


def test_vae_training_reduces_loss():
    dataset = make_dataset(n_days=24)
    vae = DailyVAE(latent_dim=2, hidden=(32, 16), seed=0).fit(
        dataset, epochs=40, batch_size=24
    )
    start = vae.history[0]["mse"] + vae.history[0]["kl"]
    end = vae.history[-1]["mse"] + vae.history[-1]["kl"]
    assert end < start
    assert len(vae.history) == 40


def test_vae_sample_contract():
    dataset = make_dataset(n_days=12)
    vae = DailyVAE(latent_dim=2, hidden=(16, 8), seed=1).fit(dataset, epochs=2, batch_size=6)
    days = vae.sample(10, dataset)
    assert len(days) == 10
    for day in days:
        assert day.k.shape == (144,)
        assert day.k.dtype == np.float32
        assert np.isfinite(day.k).all()  # raw output, never clipped
        assert np.isin(np.datetime64(day.date, "D"), dataset.dates)
        assert np.isfinite(day.irradiance()).all()
    noisy = vae.sample(5, dataset, include_noise=True)
    assert all(np.isfinite(day.k).all() for day in noisy)
    with pytest.raises(ValueError, match="empty"):
        vae.sample(3, DailyDataset(
            dates=dataset.dates[:0], k=dataset.k[:0], valid=dataset.valid[:0],
            g_clear_cal=dataset.g_clear_cal[:0], include_day=dataset.include_day[:0],
            split_id=dataset.split_id[:0],
        ))


def test_vae_save_load_roundtrip(tmp_path):
    dataset = make_dataset(n_days=12)
    vae = DailyVAE(latent_dim=2, hidden=(16, 8), seed=1).fit(dataset, epochs=2, batch_size=6)
    path = tmp_path / "vae.pt"
    vae.save(path)
    loaded = DailyVAE.load(path, seed=1)
    z = torch.zeros(1, 2)
    np.testing.assert_allclose(
        loaded._model.decode(z).detach().numpy(),
        vae._model.decode(z).detach().numpy(),
        atol=1e-6,
    )
    for day in loaded.sample(4, dataset):
        assert np.isfinite(day.k).all()
