# -*- coding: utf-8 -*-
"""
================================================================================
steps_cost_estimator.py
--------------------------------------------------------------------------------
Point estimate and 95% CI of 6-month healthcare expenditure from daily steps
(monthly average) and age.

Model (identical to Eq. (3) in Kato 2026, SSM - Population Health 35:101958)
    log(Y6 + 1) = a + s1(Steps) + s2(Age) + s3(BMI) + s4(BaselineCost6)
                  + b'[Sex, HealthCheck, COVID19] + e
    - LinearGAM (Gaussian, identity link); n_splines = 10 for the step smooth,
      pyGAM defaults (n_splines = 20) for the other smooths; default penalties.
    - Outcome: expenditure accumulated over the 6 months after the index month
      (primary horizon, chosen a priori).
    - Missing BMI imputed with the median of individual-level mean BMI (22.7).
    - Analysis period 2020-06 to 2023-06.

Two training variants (same specification and data; only the sample differs)
    'paper'     : the paper's own sample (338,706 person-months, 19,599 individuals).
                  Reproduces the published values (local nadir 10,893 steps/day,
                  global minimum 19,674 steps/day, explained deviance 0.326).
                  Note: in the source data, Age is the age in 2023 and is fixed
                  within each person. After a person turns 75 their insurance moves
                  from National Health Insurance to the Medical Care System for the
                  Elderly and the linked claims are recorded as zero, so the age
                  effect breaks down above 74. Predictions are therefore limited
                  to ages 40-74.
    'corrected' : the same specification on a cleaned sample (default for the
                  Python estimator): (i) age converted to the age at the index
                  month (= Age - (2023 - year)) and restricted to 40+; (ii)
                  person-months whose 6-month outcome window ends at age 75 or
                  older are excluded. Used for validation; the public page uses
                  the 'paper' variant.

Prediction (as in Fig. 2 / Fig. 4 of the paper)
    - Covariates that are not supplied are held at the training-sample median
      (continuous) or mode (categorical).
    - The point estimate is the log-scale prediction back-transformed with
      exp(.) - 1, i.e. the same quantity as the 17,170 JPY reported in the paper.
      It is a geometric-mean-type quantity and is far below the arithmetic mean.
    - 95% CI: 'model' = pyGAM point-wise interval on the log scale, back-transformed
      (as in Fig. 2); 'bootstrap' = percentile interval from an individual-level
      cluster bootstrap (only if the model was fitted with --bootstrap N).

Usage
    # Fit once (author environment; requires the linked person-month panel)
    python steps_cost_estimator.py fit --data data/steps_claims_panel.csv --out steps_cost_gam.pkl \
        [--bootstrap 200 --n-jobs 2]

    # Predict from the fitted model
    python steps_cost_estimator.py predict --model steps_cost_gam.pkl --steps 8000 --age 68
    python steps_cost_estimator.py predict --model steps_cost_gam.pkl --steps 8000 --age 68 --variant paper

    # Predict from the public fitted-surface grid (no pygam, no model file)
    python steps_cost_estimator.py predict --surface data/fitted_surface_grid.json --steps 8000 --age 68

    # Batch: CSV with columns steps, age [, sex, bmi, health_check, past_cost_6m, covid]
    python steps_cost_estimator.py predict --model steps_cost_gam.pkl --csv inputs.csv --out predictions.csv

    # From Python
    from steps_cost_estimator import StepsCostEstimator, predict_from_surface
    est = StepsCostEstimator.load("steps_cost_gam.pkl")
    est.predict(steps=8000, age=68)                               # corrected variant, model CI
    est.predict(steps=[5000, 11000], age=[55, 72], ci_method="bootstrap")
    est.predict(steps=10893, age=68, variant="paper")             # published value
    predict_from_surface("data/fitted_surface_grid.json", steps=8000, age=68)
================================================================================
"""
from __future__ import annotations

import argparse
import json
import pickle
import platform
import time
import warnings
from dataclasses import dataclass, field, asdict
from typing import Optional

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

try:
    import pygam
    from pygam import LinearGAM, s, f
except ImportError as e:  # pragma: no cover
    raise ImportError("pygam is required for fitting and model-based prediction: pip install pygam") from e


# ==========================================================================
# Specification (identical to the paper; do not change)
# ==========================================================================
LAG = 6                                   # primary horizon in months (a priori)
N_SPLINES_STEPS = 10                      # basis dimension of the step smooth
ANALYSIS_START = "2020-06-01"
ANALYSIS_END = "2023-06-01"
STEPS_UPPER_PCTL = 0.995                  # plotting / prediction limit for steps (99.5th percentile)
DENSE_PCTL = (0.025, 0.975)               # data-dense range used for interpretation
BMI_IMPUTE_DECIMALS = 1                   # imputation value = median of individual mean BMI (22.7)
AGE_REFERENCE_YEAR = 2023                 # year in which the Age column is defined
ELDERLY_SWITCH_AGE = 75                   # age at which NHI coverage ends (Medical Care System for the Elderly)

FEATURE_NAMES = ["Steps", "Age", "Sex", "BMI", "HealthCheck", "PastCost_6m_Sum", "COVID19"]
FACTOR_NAMES = {"Sex", "HealthCheck", "COVID19"}
TARGET_COL = f"FutureCost_{LAG}m_Sum"

VARIANTS = {
    "paper": dict(label="paper's own sample (reproduces the published values)", age_mode="reference",
                  exclude_post_switch=False, min_age=None, valid_age_max=74),
    "corrected": dict(label="post-75 person-months excluded, age at the index month (default for prediction)",
                      age_mode="index_month", exclude_post_switch=True, min_age=40, valid_age_max=None),
}
DEFAULT_VARIANT = "corrected"

# input names (predict() arguments / batch-CSV columns) -> feature names
INPUT_ALIASES = {
    "steps": "Steps", "age": "Age", "sex": "Sex", "bmi": "BMI",
    "health_check": "HealthCheck", "past_cost_6m": "PastCost_6m_Sum", "covid": "COVID19",
}


def _build_terms():
    # column order is fixed by FEATURE_NAMES: Steps, Age, Sex, BMI, HealthCheck, PastCost, COVID19
    return (s(0, n_splines=N_SPLINES_STEPS) + s(1) + f(2) + s(3) + f(4) + s(5) + f(6))


# ==========================================================================
# Pre-processing (same logic as the paper's analysis code)
# ==========================================================================
def load_and_engineer(path_or_df, bmi_col: str = "BMI 2") -> tuple[pd.DataFrame, float]:
    """
    Read the person-month panel, build PastCost_6m_Sum / FutureCost_6m_Sum and impute BMI.
    bmi_col: BMI column with missing values coded as NaN ('BMI 2' in the source panel; the older
             'BMI' column codes missing as 0 and is used as a fallback).
    Returns (df, bmi_impute_value).
    """
    if isinstance(path_or_df, pd.DataFrame):
        df = path_or_df.copy()
    else:
        try:
            df = pd.read_csv(path_or_df, encoding="utf-8-sig")
        except UnicodeDecodeError:
            df = pd.read_csv(path_or_df, encoding="shift_jis")

    if bmi_col not in df.columns:
        bmi_col = "BMI"
        df[bmi_col] = df[bmi_col].replace(0, np.nan)

    df["Date"] = pd.to_datetime(df["YearMonth"], format="%Y/%m", errors="coerce")
    if df["Date"].isna().any():
        df["Date"] = pd.to_datetime(df["YearMonth"])
    df = df.sort_values(["uuid", "Date"]).reset_index(drop=True)

    df = df.dropna(subset=["Age"]).copy()
    df["Cost"] = df["Cost"].fillna(0)

    for col in FACTOR_NAMES:
        if not np.issubdtype(df[col].dtype, np.number):
            df[col] = df[col].astype("category").cat.codes
        elif df[col].isna().any():
            df[col] = df[col].fillna(df[col].mode()[0])

    # BMI: impute with the median of individual-level mean BMI (22.7 in the paper)
    bmi_impute = round(float(df.groupby("uuid")[bmi_col].mean().median()), BMI_IMPUTE_DECIMALS)
    df["BMI_raw"] = df[bmi_col]
    df["BMI"] = df[bmi_col].fillna(bmi_impute)

    # baseline: previous 6 months; outcome: following 6 months (index month excluded from both)
    df["PastCost_6m_Sum"] = df.groupby("uuid")["Cost"].transform(
        lambda x: x.shift(1).rolling(window=6, min_periods=6).sum())
    indexer = pd.api.indexers.FixedForwardWindowIndexer(window_size=LAG)
    df[TARGET_COL] = df.groupby("uuid")["Cost"].transform(
        lambda x: x.shift(-1).rolling(window=indexer).sum())
    return df, bmi_impute


def slice_analysis(df: pd.DataFrame, start=ANALYSIS_START, end=ANALYSIS_END) -> pd.DataFrame:
    """Restrict to the analysis period and to rows with a complete baseline and outcome window."""
    d = df[(df["Date"] >= pd.Timestamp(start)) & (df["Date"] <= pd.Timestamp(end))].copy()
    d = d.dropna(subset=[TARGET_COL, "PastCost_6m_Sum", "Steps"] + FEATURE_NAMES).copy()
    return d.reset_index(drop=True)


def apply_variant(d: pd.DataFrame, variant: str) -> pd.DataFrame:
    """Apply the sample definition of a variant; the 'Age' column becomes the age used for fitting."""
    v = VARIANTS[variant]
    d = d.copy()
    d["Age_ref"] = d["Age"]                                    # age in the reference year (source column)
    year_t = d["Date"].dt.year
    d["Age_t"] = d["Age_ref"] - (AGE_REFERENCE_YEAR - year_t)  # age at the index month (year resolution)
    if v["exclude_post_switch"]:
        year_end = (d["Date"] + pd.DateOffset(months=LAG)).dt.year
        age_end = d["Age_ref"] - (AGE_REFERENCE_YEAR - year_end)
        d = d[age_end < ELDERLY_SWITCH_AGE].copy()
    if v["age_mode"] == "index_month":
        d["Age"] = d["Age_t"]
    if v["min_age"] is not None:
        d = d[d["Age"] >= v["min_age"]].copy()
    return d.reset_index(drop=True)


# ==========================================================================
# Estimator
# ==========================================================================
@dataclass
class FitInfo:
    variant: str = ""
    variant_label: str = ""
    n_person_months: int = 0
    n_individuals: int = 0
    analysis_start: str = ANALYSIS_START
    analysis_end: str = ANALYSIS_END
    lag_months: int = LAG
    n_splines_steps: int = N_SPLINES_STEPS
    bmi_impute_value: float = np.nan
    bmi_missing_share: float = np.nan
    pseudo_r2: float = np.nan
    aic: float = np.nan
    gcv: float = np.nan
    edof_total: float = np.nan
    steps_upper: float = np.nan            # 99.5th percentile (prediction limit)
    steps_dense: tuple = (np.nan, np.nan)  # 2.5th-97.5th percentile (interpretable range)
    age_range: tuple = (np.nan, np.nan)    # age range of the training sample
    age_valid: tuple = (np.nan, np.nan)    # age range accepted for prediction
    fixed_covariates: dict = field(default_factory=dict)
    local_nadir: dict = field(default_factory=dict)
    global_min: dict = field(default_factory=dict)
    n_bootstrap: int = 0
    bootstrap_seed: Optional[int] = None


@dataclass
class _SubModel:
    gam: LinearGAM
    info: FitInfo
    boot_models: list = field(default_factory=list)


class StepsCostEstimator:
    """Daily steps x age -> 6-month healthcare expenditure (point estimate + 95% CI)."""

    def __init__(self):
        self.models: dict[str, _SubModel] = {}
        self.meta: dict = {}

    # ---------------------------------------------------------------- fit
    def fit(self, data, variants=("paper", "corrected"), bmi_col: str = "BMI 2",
            n_bootstrap: int = 0, bootstrap_variants=(DEFAULT_VARIANT,), seed: int = 42,
            n_jobs: int = 1, verbose: bool = True) -> "StepsCostEstimator":
        t0 = time.time()
        df, bmi_impute = load_and_engineer(data, bmi_col=bmi_col)
        d_all = slice_analysis(df)
        self.meta = dict(
            versions=dict(python=platform.python_version(), numpy=np.__version__,
                          pandas=pd.__version__, pygam=pygam.__version__),
            fitted_at=pd.Timestamp.now().strftime("%Y-%m-%d %H:%M"),
            data_rows=int(len(df)), bmi_col=bmi_col,
        )
        for variant in variants:
            d = apply_variant(d_all, variant)
            if verbose:
                print(f"\n[fit:{variant}] {VARIANTS[variant]['label']}")
                print(f"[fit:{variant}] analytic sample: {len(d):,} person-months / "
                      f"{d['uuid'].nunique():,} individuals, age {d['Age'].min():.0f}-{d['Age'].max():.0f}")
            sub = self._fit_one(d, variant, bmi_impute)
            self.models[variant] = sub
            if verbose:
                i = sub.info
                print(f"[fit:{variant}] explained deviance = {i.pseudo_r2:.4f}, AIC = {i.aic:,.0f}, "
                      f"EDoF = {i.edof_total:.1f}  ({time.time()-t0:.0f}s)")
                if i.local_nadir:
                    ln = i.local_nadir
                    print(f"[fit:{variant}] local nadir: {ln['steps']:,.0f} steps -> {ln['cost']:,.0f} JPY "
                          f"[{ln['lower']:,.0f}-{ln['upper']:,.0f}] (age fixed at {i.fixed_covariates['Age']:.0f})")
                else:
                    print(f"[fit:{variant}] no strict local minimum inside the plotted range")
                gm = i.global_min
                print(f"[fit:{variant}] global minimum: {gm['steps']:,.0f} steps -> {gm['cost']:,.0f} JPY")
            if n_bootstrap > 0 and variant in bootstrap_variants:
                self._fit_bootstrap(sub, d, n_bootstrap, seed, n_jobs, verbose)
        return self

    def _fit_one(self, d: pd.DataFrame, variant: str, bmi_impute: float) -> _SubModel:
        X = d[FEATURE_NAMES].values.astype(float)
        y = np.log1p(d[TARGET_COL].values.astype(float))
        gam = LinearGAM(_build_terms()).fit(X, y)
        st = gam.statistics_

        fixed = {}
        for j, name in enumerate(FEATURE_NAMES):
            col = X[:, j]
            fixed[name] = float(pd.Series(col).mode()[0]) if name in FACTOR_NAMES else float(np.median(col))

        v = VARIANTS[variant]
        age_min, age_max = float(d["Age"].min()), float(d["Age"].max())
        valid_max = age_max if v["valid_age_max"] is None else min(age_max, float(v["valid_age_max"]))
        info = FitInfo(
            variant=variant, variant_label=v["label"],
            n_person_months=int(len(d)), n_individuals=int(d["uuid"].nunique()),
            bmi_impute_value=bmi_impute, bmi_missing_share=float(d["BMI_raw"].isna().mean()),
            pseudo_r2=float(st["pseudo_r2"]["explained_deviance"]),
            aic=float(st["AIC"]), gcv=float(st["GCV"]), edof_total=float(st["edof"]),
            steps_upper=float(d["Steps"].quantile(STEPS_UPPER_PCTL)),
            steps_dense=(float(d["Steps"].quantile(DENSE_PCTL[0])), float(d["Steps"].quantile(DENSE_PCTL[1]))),
            age_range=(age_min, age_max), age_valid=(age_min, valid_max),
            fixed_covariates=fixed,
        )
        sub = _SubModel(gam=gam, info=info)
        info.local_nadir, info.global_min = self._find_optima(sub)
        return sub

    def _fit_bootstrap(self, sub: _SubModel, d: pd.DataFrame, B: int, seed: int, n_jobs: int, verbose: bool):
        """Individual-level (uuid) cluster bootstrap, resampled as in the paper's bootstrap analysis."""
        idx_by_id = {i: g.index.values for i, g in d.groupby("uuid")}
        ids = np.array(list(idx_by_id.keys()))
        Xall = d[FEATURE_NAMES].values.astype(float)
        yall = np.log1p(d[TARGET_COL].values.astype(float))
        n_levels = {j: len(np.unique(Xall[:, j])) for j, nm in enumerate(FEATURE_NAMES) if nm in FACTOR_NAMES}

        def one(child_seed):
            rng = np.random.default_rng(child_seed)
            samp = rng.choice(ids, size=len(ids), replace=True)
            rows = np.concatenate([idx_by_id[i] for i in samp])
            Xb, yb = Xall[rows], yall[rows]
            for j, k in n_levels.items():            # drop the rare resample that loses a factor level
                if len(np.unique(Xb[:, j])) != k:
                    return None
            return LinearGAM(_build_terms()).fit(Xb, yb)

        seeds = np.random.SeedSequence(seed).spawn(B)
        t0 = time.time()
        if n_jobs > 1:
            from joblib import Parallel, delayed
            models = Parallel(n_jobs=n_jobs, backend="loky", verbose=5 if verbose else 0)(
                delayed(one)(sd) for sd in seeds)
        else:
            models = []
            for b, sd in enumerate(seeds, 1):
                models.append(one(sd))
                if verbose and (b % 10 == 0 or b == B):
                    print(f"  bootstrap {b}/{B}  ({time.time()-t0:.0f}s)")
        sub.boot_models = [m for m in models if m is not None]
        sub.info.n_bootstrap = len(sub.boot_models)
        sub.info.bootstrap_seed = seed
        if verbose:
            print(f"[fit:{sub.info.variant}] cluster bootstrap: {len(sub.boot_models)} valid replicates "
                  f"({time.time()-t0:.0f}s)")

    # ------------------------------------------------------------ predict
    def _get(self, variant: Optional[str]) -> _SubModel:
        if not self.models:
            raise RuntimeError("No fitted model: call fit() or load() first")
        variant = variant or DEFAULT_VARIANT
        if variant not in self.models:
            raise ValueError(f"variant '{variant}' is not fitted (available: {list(self.models)})")
        return self.models[variant]

    def _design(self, sub: _SubModel, steps, age, sex=None, bmi=None, health_check=None,
                past_cost_6m=None, covid=None, extrapolation: str = "clip"):
        info = sub.info
        steps = np.atleast_1d(np.asarray(steps, dtype=float))
        age = np.atleast_1d(np.asarray(age, dtype=float))
        n = max(len(steps), len(age))
        if len(steps) == 1: steps = np.repeat(steps, n)
        if len(age) == 1: age = np.repeat(age, n)
        if len(steps) != len(age):
            raise ValueError("steps and age must have the same length")

        fx = info.fixed_covariates
        opt = {"Sex": sex, "BMI": bmi, "HealthCheck": health_check,
               "PastCost_6m_Sum": past_cost_6m, "COVID19": covid}
        cols = {"Steps": steps.copy(), "Age": age.copy()}
        for name, val in opt.items():
            if val is None:
                cols[name] = np.full(n, fx[name])
            else:
                v = np.atleast_1d(np.asarray(val, dtype=float))
                v = np.repeat(v, n) if len(v) == 1 else v
                cols[name] = np.where(np.isnan(v), fx[name], v)

        lo_s, hi_s = 0.0, info.steps_upper
        lo_a, hi_a = info.age_valid
        out_steps = (cols["Steps"] < lo_s) | (cols["Steps"] > hi_s)
        out_age = (cols["Age"] < lo_a) | (cols["Age"] > hi_a)
        if extrapolation == "raise" and (out_steps.any() or out_age.any()):
            raise ValueError(f"input outside the fitted range (steps {lo_s:.0f}-{hi_s:.0f}, age {lo_a:.0f}-{hi_a:.0f})")
        if extrapolation == "clip":
            cols["Steps"] = np.clip(cols["Steps"], lo_s, hi_s)
            cols["Age"] = np.clip(cols["Age"], lo_a, hi_a)
        elif extrapolation != "allow":
            raise ValueError("extrapolation must be 'clip', 'allow' or 'raise'")

        X = np.column_stack([cols[nm] for nm in FEATURE_NAMES])
        dense_lo, dense_hi = info.steps_dense
        meta = pd.DataFrame({
            "steps_input": steps, "age_input": age,
            "steps_used": cols["Steps"], "age_used": cols["Age"],
            "in_dense_range": (cols["Steps"] >= dense_lo) & (cols["Steps"] <= dense_hi),
            "clipped": (out_steps | out_age) if extrapolation == "clip" else np.zeros(n, bool),
        })
        return X, meta

    def predict(self, steps, age, sex=None, bmi=None, health_check=None, past_cost_6m=None,
                covid=None, ci: float = 0.95, ci_method: str = "model",
                variant: Optional[str] = None, extrapolation: str = "clip") -> pd.DataFrame:
        """
        Parameters
        ----------
        steps : daily steps, monthly average (scalar or array)
        age   : age in years (scalar or array)
        sex, bmi, health_check, past_cost_6m, covid :
            optional covariates; when omitted they are held at the training-sample median/mode
            (as in Fig. 2 of the paper). sex: 1 = male, 2 = female; health_check: 0/1;
            past_cost_6m: expenditure over the previous 6 months [JPY]; covid: 0/1 (state-of-emergency month)
        ci        : confidence level (default 0.95)
        ci_method : 'model' (pyGAM point-wise interval, as in Fig. 2) or
                    'bootstrap' (percentile interval from the individual-level cluster bootstrap)
        variant   : 'corrected' (default) or 'paper'
        extrapolation : 'clip' (default: inputs outside the fitted range are moved to the boundary),
                        'allow' or 'raise'

        Returns
        -------
        DataFrame with columns steps_input, age_input, steps_used, age_used, point, lower, upper,
        in_dense_range, clipped, variant, ci_method, ci. Values are JPY over 6 months; `point` is the
        back-transformed log-model prediction (geometric-mean type).
        """
        if not (0 < ci < 1):
            raise ValueError("ci must be in (0, 1)")
        sub = self._get(variant)
        X, meta = self._design(sub, steps, age, sex, bmi, health_check, past_cost_6m, covid, extrapolation)

        pred_log = sub.gam.predict(X)
        if ci_method == "model":
            band = sub.gam.confidence_intervals(X, width=ci)
            lo_log, hi_log = band[:, 0], band[:, 1]
        elif ci_method == "bootstrap":
            if not sub.boot_models:
                raise RuntimeError(f"variant '{sub.info.variant}' has no bootstrap replicates "
                                   "(fit with --bootstrap N)")
            P = np.vstack([m.predict(X) for m in sub.boot_models])        # (B, n)
            a = (1 - ci) / 2
            lo_log, hi_log = np.percentile(P, 100 * a, axis=0), np.percentile(P, 100 * (1 - a), axis=0)
        else:
            raise ValueError("ci_method must be 'model' or 'bootstrap'")

        out = meta.copy()
        out["point"] = np.round(np.expm1(pred_log), 0)
        out["lower"] = np.round(np.expm1(lo_log), 0)
        out["upper"] = np.round(np.expm1(hi_log), 0)
        out["variant"] = sub.info.variant
        out["ci_method"] = ci_method
        out["ci"] = ci
        return out

    def predict_grid(self, steps_step: float = 100, age_step: float = 1, variant: Optional[str] = None,
                     **kw) -> pd.DataFrame:
        """Predict on a steps x age grid (used to build the fitted-surface grid)."""
        info = self._get(variant).info
        sg = np.arange(0, info.steps_upper + 1e-9, steps_step)
        ag = np.arange(np.ceil(info.age_valid[0]), np.floor(info.age_valid[1]) + 1e-9, age_step)
        S, A = np.meshgrid(sg, ag, indexing="ij")
        return self.predict(S.ravel(), A.ravel(), variant=variant, **kw)

    # ---------------------------------------------------------- diagnostics
    def _find_optima(self, sub: _SubModel, n_grid: int = 200, order: int = 5) -> tuple[dict, dict]:
        """Local and global minima of the step curve on the paper's grid (0 to 99.5th pct, 200 points)."""
        from scipy.signal import argrelextrema
        info = sub.info
        grid = np.linspace(0, info.steps_upper, n_grid)
        X, _ = self._design(sub, grid, info.fixed_covariates["Age"], extrapolation="allow")
        pl = sub.gam.predict(X)
        band = sub.gam.confidence_intervals(X, width=0.95)
        loc = argrelextrema(pl, np.less, order=order)[0]
        g = int(np.argmin(pl))
        pack = lambda i: dict(steps=float(grid[i]), cost=float(np.expm1(pl[i])),
                              lower=float(np.expm1(band[i, 0])), upper=float(np.expm1(band[i, 1])))
        local = pack(loc[0]) if len(loc) else {}
        return local, pack(g)

    def summary(self, variant: Optional[str] = None) -> dict:
        if variant is None:
            return {"meta": self.meta, "variants": {k: asdict(v.info) for k, v in self.models.items()}}
        return asdict(self._get(variant).info)

    # ---------------------------------------------------------------- io
    def save(self, path: str):
        payload = {"meta": self.meta, "feature_names": FEATURE_NAMES, "models": {}}
        for k, sub in self.models.items():
            payload["models"][k] = {"gam": sub.gam, "boot_models": sub.boot_models, "info": asdict(sub.info)}
        with open(path, "wb") as fh:
            pickle.dump(payload, fh)

    @classmethod
    def load(cls, path: str) -> "StepsCostEstimator":
        with open(path, "rb") as fh:
            obj = pickle.load(fh)
        est = cls()
        est.meta = obj.get("meta", {})
        for k, m in obj["models"].items():
            info = m["info"]
            for key in ("steps_dense", "age_range", "age_valid"):
                info[key] = tuple(info[key])
            est.models[k] = _SubModel(gam=m["gam"], info=FitInfo(**info), boot_models=m.get("boot_models", []))
        return est


# ==========================================================================
# Prediction from the public fitted-surface grid (no model file, no pygam)
# ==========================================================================
def predict_from_surface(surface_json, steps, age) -> pd.DataFrame:
    """
    Point estimate and 95% CI from data/fitted_surface_grid.json (paper variant, 100-step x 1-year grid)
    by bilinear interpolation on the log scale. Inputs outside the grid (steps 0-20,000, age 40-74)
    are moved to the boundary and flagged in `clipped`.
    """
    with open(surface_json, encoding="utf-8") as fh:
        S = json.load(fh)
    sg = np.asarray(S["steps"], float); ag = np.asarray(S["ages"], float)
    piv = {k: np.asarray(S[k], float) for k in ("point", "lower", "upper")}      # [age][step]
    dense_lo, dense_hi = S["meta"]["steps_dense"]
    steps = np.atleast_1d(np.asarray(steps, float)); age = np.atleast_1d(np.asarray(age, float))
    n = max(len(steps), len(age))
    steps = np.repeat(steps, n) if len(steps) == 1 else steps
    age = np.repeat(age, n) if len(age) == 1 else age
    s_ = np.clip(steps, sg[0], sg[-1]); a_ = np.clip(age, ag[0], ag[-1])
    i = np.clip(np.searchsorted(sg, s_, side="right") - 1, 0, len(sg) - 2)
    j = np.clip(np.searchsorted(ag, a_, side="right") - 1, 0, len(ag) - 2)
    ts = (s_ - sg[i]) / (sg[i + 1] - sg[i]); ta = (a_ - ag[j]) / (ag[j + 1] - ag[j])
    out = pd.DataFrame({"steps_input": steps, "age_input": age, "steps_used": s_, "age_used": a_,
                        "in_dense_range": (s_ >= dense_lo) & (s_ <= dense_hi),
                        "clipped": (steps != s_) | (age != a_)})
    for k, z in piv.items():
        z = np.log1p(z)
        v = (1 - ta) * ((1 - ts) * z[j, i] + ts * z[j, i + 1]) + ta * ((1 - ts) * z[j + 1, i] + ts * z[j + 1, i + 1])
        out[k] = np.round(np.expm1(v), 0)
    out["variant"] = S["meta"].get("variant", "paper"); out["ci_method"] = "model"; out["ci"] = 0.95
    return out


# ==========================================================================
# CLI
# ==========================================================================
def _print_one(r, ci: float, label: str, info: Optional[FitInfo] = None):
    print(f"steps={r.steps_used:,.0f}/day, age={r.age_used:.0f} -> 6-month expenditure {r.point:,.0f} JPY  "
          f"{int(ci*100)}% CI [{r.lower:,.0f} - {r.upper:,.0f}]  ({label})")
    if r.clipped:
        rng = (f" (steps 0-{info.steps_upper:,.0f}, age {info.age_valid[0]:.0f}-{info.age_valid[1]:.0f})"
               if info else "")
        print(f"  note: input outside the fitted range{rng}; moved to the boundary")
    if not r.in_dense_range:
        rng = f" ({info.steps_dense[0]:,.0f}-{info.steps_dense[1]:,.0f} steps/day)" if info else ""
        print(f"  note: steps outside the data-dense range{rng}; treat as extrapolation")


def _cli():
    ap = argparse.ArgumentParser(description="Daily steps x age -> 6-month healthcare expenditure (point estimate + 95% CI)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    pf = sub.add_parser("fit", help="fit the model (requires the linked person-month panel)")
    pf.add_argument("--data", required=True, help="person-month panel CSV (see data/schema.csv)")
    pf.add_argument("--out", default="steps_cost_gam.pkl")
    pf.add_argument("--bmi-col", default="BMI 2")
    pf.add_argument("--variants", nargs="+", default=["paper", "corrected"], choices=list(VARIANTS))
    pf.add_argument("--bootstrap", type=int, default=0, help="number of cluster-bootstrap replicates (0 = none)")
    pf.add_argument("--bootstrap-variants", nargs="+", default=[DEFAULT_VARIANT], choices=list(VARIANTS))
    pf.add_argument("--seed", type=int, default=42)
    pf.add_argument("--n-jobs", type=int, default=1)
    pf.add_argument("--grid-csv", default=None, help="also write a steps x age prediction grid (default variant) to this CSV")

    pp = sub.add_parser("predict", help="predict from a fitted model or from the fitted-surface grid")
    pp.add_argument("--model", default="steps_cost_gam.pkl")
    pp.add_argument("--surface", default=None,
                    help="predict from data/fitted_surface_grid.json instead of a model file (paper variant, model CI)")
    pp.add_argument("--steps", type=float)
    pp.add_argument("--age", type=float)
    pp.add_argument("--sex", type=float); pp.add_argument("--bmi", type=float)
    pp.add_argument("--health-check", type=float); pp.add_argument("--past-cost-6m", type=float)
    pp.add_argument("--covid", type=float)
    pp.add_argument("--csv", help="batch input CSV with columns steps, age [, sex, bmi, health_check, past_cost_6m, covid]")
    pp.add_argument("--out", help="output CSV for batch prediction")
    pp.add_argument("--ci", type=float, default=0.95)
    pp.add_argument("--ci-method", choices=["model", "bootstrap"], default="model")
    pp.add_argument("--variant", choices=list(VARIANTS), default=DEFAULT_VARIANT)
    pp.add_argument("--extrapolation", choices=["clip", "allow", "raise"], default="clip")

    pi = sub.add_parser("info", help="print the fitted model summary as JSON")
    pi.add_argument("--model", default="steps_cost_gam.pkl")

    a = ap.parse_args()
    if a.cmd == "fit":
        est = StepsCostEstimator().fit(a.data, variants=a.variants, bmi_col=a.bmi_col,
                                       n_bootstrap=a.bootstrap, bootstrap_variants=a.bootstrap_variants,
                                       seed=a.seed, n_jobs=a.n_jobs)
        est.save(a.out)
        print(f"\n[fit] saved -> {a.out}")
        if a.grid_csv:
            g = est.predict_grid(variant=DEFAULT_VARIANT)
            if est.models[DEFAULT_VARIANT].boot_models:
                gb = est.predict_grid(variant=DEFAULT_VARIANT, ci_method="bootstrap")
                g["lower_boot"], g["upper_boot"] = gb["lower"], gb["upper"]
            g.to_csv(a.grid_csv, index=False)
            print(f"[fit] prediction grid -> {a.grid_csv} ({len(g):,} rows)")
    elif a.cmd == "predict" and a.surface:
        if a.csv:
            df = pd.read_csv(a.csv)
            if "steps" not in df.columns or "age" not in df.columns:
                ap.error("the batch CSV needs columns 'steps' and 'age'")
            res = predict_from_surface(a.surface, df["steps"].values, df["age"].values)
            if a.out:
                res.to_csv(a.out, index=False); print(f"-> {a.out} ({len(res):,} rows)")
            else:
                print(res.to_string(index=False))
        else:
            if a.steps is None or a.age is None:
                ap.error("give --steps and --age (or --csv)")
            _print_one(predict_from_surface(a.surface, a.steps, a.age).iloc[0], 0.95, "fitted-surface grid, paper variant")
    elif a.cmd == "predict":
        est = StepsCostEstimator.load(a.model)
        kw = dict(ci=a.ci, ci_method=a.ci_method, variant=a.variant, extrapolation=a.extrapolation)
        if a.csv:
            df = pd.read_csv(a.csv)
            cols = {k: df[k].values for k in INPUT_ALIASES if k in df.columns}
            if "steps" not in cols or "age" not in cols:
                ap.error("the batch CSV needs columns 'steps' and 'age'")
            res = est.predict(**cols, **kw)
            if a.out:
                res.to_csv(a.out, index=False); print(f"-> {a.out} ({len(res):,} rows)")
            else:
                print(res.to_string(index=False))
        else:
            if a.steps is None or a.age is None:
                ap.error("give --steps and --age (or --csv)")
            res = est.predict(a.steps, a.age, sex=a.sex, bmi=a.bmi, health_check=a.health_check,
                              past_cost_6m=a.past_cost_6m, covid=a.covid, **kw)
            _print_one(res.iloc[0], a.ci, f"variant={a.variant}, CI={a.ci_method}", est.models[a.variant].info)
    elif a.cmd == "info":
        est = StepsCostEstimator.load(a.model)
        print(json.dumps(est.summary(), ensure_ascii=False, indent=2, default=float))


if __name__ == "__main__":
    _cli()
