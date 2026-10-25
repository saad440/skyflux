# SkyFlux

**Physics-guided modelling of cloud-driven solar irradiance fluctuations.**

SkyFlux investigates whether lightweight generative models can reproduce both the distribution of solar irradiance and its short-term variability. It uses 10-minute observations from the German Weather Service (DWD) at Belm, Germany (2020-2025).

The approach separates predictable clear-sky irradiance from a variable cloud factor:

$$
G(t) = G_{\mathrm{clear,cal}}(t)\,k(t).
$$

A calibrated `pvlib` clear-sky reference accounts for the daily and seasonal solar cycle; the generators model the remaining daylight fluctuations.

## Notebooks

- [01. From sunlight measurements to cloud fluctuations](notebooks/01_physical_representation.ipynb): the data and physical representation.
- [02. What do the generators reproduce — and miss?](notebooks/02_generative_comparison.ipynb): historical-day bootstrap, probabilistic PCA, and a variational autoencoder.

Models are trained on 2020-2023, validated on 2024, and evaluated on 2025.

**Results:** the tested learned generators substantially underestimate short-term fluctuations. Adding independent noise improves ramp magnitudes, but not the overall temporal behaviour.

## Run locally

With Python 3.11 or newer:

```bash
python -m pip install -e . jupyterlab
jupyter lab
```

The notebooks use included compact assets, so no raw DWD download is needed to explore the results. Data-processing and training scripts are in `scripts/`.

**Data source:** [DWD Climate Data Center - 10-minute solar observations](https://opendata.dwd.de/climate_environment/CDC/observations_germany/climate/10_minutes/solar/historical/).
