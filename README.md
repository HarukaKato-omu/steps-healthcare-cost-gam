# Daily steps and 6-month healthcare expenditure

Estimator and web tool for the generalized additive model (GAM) in:

> Kato, H. (2026). *Non-linear association between mHealth-measured daily steps and subsequent
> healthcare expenditures among long-term app users in Osaka, Japan.* **SSM – Population Health**,
> 35, 101958. https://doi.org/10.1016/j.ssmph.2026.101958

Given **daily steps (monthly average)** and **age**, it returns the model's point estimate and 95% CI
of healthcare expenditure over the following 6 months, and shows the point on the fitted steps × age
surface (Figure 4 of the article).

**Web tool:** https://harukakato-omu.github.io/steps-healthcare-cost-gam/

The individual-level data are not redistributed. The repository contains the model specification and
training code, the fitted surface on a 100-step × 1-year grid (aggregate output of the model, nothing
else), and the static page that reads it.

## Repository structure

```
├── src/
│   └── steps_cost_estimator.py   # Model specification, training, prediction (model file or grid), CLI
├── scripts/
│   └── build_site.py             # Writes data/fitted_surface_grid.json and docs/index.html from a fitted model
├── docs/
│   ├── template.html             # Page source
│   └── index.html                # Generated page served by GitHub Pages (grid embedded)
├── data/
│   ├── schema.csv                # Layout of the person-month panel: variables, definitions, sources (no raw data)
│   └── fitted_surface_grid.json  # Fitted surface: steps 0–20,000 (step 100) × age 40–74, point + 95% CI
├── .gitignore                    # Prevents accidental data/model commits
├── requirements.txt
├── CITATION.cff
├── LICENSE
└── README.md
```

## Data availability

The person-month panel (`data/steps_claims_panel.csv` in the author's environment; layout in
`data/schema.csv`) links step counts from A-Smile, the Osaka Prefecture mHealth application, to
National Health Insurance claims for 19,599 long-term users aged 40 and over (338,706 person-months,
June 2020 – June 2023). It was provided by Osaka Prefecture for approved academic research under a
data-use agreement that prohibits re-identification and re-provision to third parties, and it is
**not** redistributed here. `data/fitted_surface_grid.json` is the only data file tracked: it holds
model predictions on a grid and cannot be used to recover any individual record.

## Model

```
log(Y6 + 1) = α + s1(Steps; k = 10) + s2(Age) + s3(BMI) + s4(Baseline6) + β·[Sex, HealthCheck, COVID-19]
```

Gaussian GAM with identity link fitted with pyGAM 0.12.0 (default penalties; 10 basis functions for
the step smooth, pyGAM defaults for the others). The outcome is expenditure accumulated over the 6
months after the index month; the baseline is the 6 months before it. Missing BMI (47.8% of
person-months) is imputed with 22.7, the median of individual-level mean BMI. Explained deviance 0.326.

The refit in this repository reproduces the published sample size, explained deviance and the
positions of the local nadir (10,893 steps/day) and global minimum (19,674 steps/day); the
expenditure level at the nadir is 1.5% higher (17,432 vs 17,170 JPY), attributable to rounding of the
fixed covariates.

## Read this before using the numbers

- **Association, not intervention effect.** The curve describes expenditure differences across people
  with different step counts in one selected cohort, adjusted for age, sex, BMI, health-check
  attendance, baseline expenditure and state-of-emergency months. It does not estimate what would
  happen if a person changed their steps.
- **Geometric-mean-type quantity.** The model is fitted on log(Y + 1); the back-transformed prediction
  is far below the arithmetic mean of 6-month expenditures. Do not sum the estimates into budget totals.
- **Point-wise 95% CI of the fitted curve** (as in Figure 2 of the article), not a prediction interval
  for an individual.
- **Ages 40–74, 0–20,000 steps/day.** In the source data age is the age in 2023 and is fixed within
  each person; at 75 enrolment moves from National Health Insurance to the Medical Care System for the
  Elderly and the linked claims end, so the model carries no valid information above 74. The article
  interprets the step curve only within the data-dense range (845–14,906 steps/day); the tool stops at
  20,000 rather than at the article's plotting limit (99.5th percentile, 22,119), and flags steps above
  14,906 as extrapolated.
- Covariates other than steps and age are held at the sample median/mode (female, BMI 22.7, no health
  check, baseline 6-month expenditure 75,220 JPY, non-emergency month), as in the article's figures.

## Requirements

Prediction from the grid needs only `numpy` and `pandas`. Refitting and rebuilding the page need
`pygam==0.12.0`, `scipy`, `joblib` (parallel bootstrap) and the person-month panel.

```bash
pip install -r requirements.txt
```

## How to run

**Web page.** Open `docs/index.html` or the GitHub Pages site. Between grid points the page
interpolates on the log scale; at grid points the values equal the model output to the yen.

**Python, from the fitted-surface grid (no model file):**

```bash
python src/steps_cost_estimator.py predict --surface data/fitted_surface_grid.json --steps 8000 --age 68
#  steps=8,000/day, age=68 -> 6-month expenditure 19,044 JPY  95% CI [18,278 - 19,843]
python src/steps_cost_estimator.py predict --surface data/fitted_surface_grid.json --csv inputs.csv --out predictions.csv
```

```python
from steps_cost_estimator import predict_from_surface
predict_from_surface("data/fitted_surface_grid.json", steps=[5000, 11000], age=[55, 72])
```

**Refit and rebuild (author environment; requires the panel):**

```bash
python src/steps_cost_estimator.py fit --data data/steps_claims_panel.csv --out steps_cost_gam.pkl --bootstrap 200 --n-jobs 2
python scripts/build_site.py --model steps_cost_gam.pkl
```

`fit` trains two variants: `paper` (the article's own sample; used for the public grid and page) and
`corrected` (person-months after the age-75 insurance switch excluded, age taken at the index month;
used for internal validation, with an individual-level cluster-bootstrap CI). `predict --model`
accepts `--variant`, `--ci-method bootstrap` and the optional covariates
`--sex --bmi --health-check --past-cost-6m --covid`.

## Reproducibility notes

- Python 3.11, numpy 2.4, pandas 3.0, pygam 0.12.0; GAM fitting is deterministic. The bootstrap uses
  `numpy.random.SeedSequence(42)`.
- `docs/index.html` is generated from `docs/template.html`; edit the template and rerun
  `scripts/build_site.py`. The page loads `plotly.js 2.35.3` from cdnjs and fonts from Google Fonts;
  everything else is embedded.

## Citation

Please cite the article above. `CITATION.cff` describes this software; GitHub shows it under
"Cite this repository".

## License

MIT for the code and the page. `data/fitted_surface_grid.json` is released under CC BY 4.0.

## Contact

Haruka Kato, Osaka Metropolitan University — haruka-kato@omu.ac.jp
