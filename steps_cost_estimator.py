# -*- coding: utf-8 -*-
"""
================================================================================
steps_cost_estimator.py
--------------------------------------------------------------------------------
月平均歩数と年齢から「今後 6 か月の総医療費」を推計する（点推定 + 95% 信頼区間）

学習モデル（Kato 2026, SSM - Population Health 35:101958, Eq.(3) と同一仕様）
    log(Y6 + 1) = α + s1(Steps) + s2(Age) + s3(BMI) + s4(BaselineCost6)
                  + β·[Sex, HealthCheck, COVID19] + ε
    - LinearGAM（恒等リンク, Gaussian）, Steps の平滑化基底 n_splines = 10,
      他の平滑化項は pyGAM 既定（n_splines = 20）, 罰則 λ は既定値（論文と同一）
    - 目的変数: index 月を除く先 6 か月の累積医療費（主要 horizon, a priori）
    - BMI 欠損は「個人別平均 BMI の中央値」(22.7) で補完（論文と同一）
    - 分析期間 2020-06 〜 2023-06

2 つの学習バリアント（同じモデル仕様・同じデータ、学習標本の定義のみ異なる）
    'paper'     : 論文と同一の標本（n = 338,706 person-months, 19,599 人）。
                  公表値（局所最小 10,893 歩, 全体最小 19,674 歩, pseudo R² 0.326）を再現する。
                  ※ データの Age は「2023 年時点の年齢」で個人内固定。75 歳到達（国保 →
                    後期高齢者医療制度）後の月は費用が 0 として記録されているため、
                    2023 年時点 75 歳以上の個人では年齢効果が崩れる（76 歳以上は使用不可）。
    'corrected' : 上記アーティファクトを除いた標本（既定）。
                  (i) 年齢を index 月時点の年齢（= Age − (2023 − 年)）に換算し 40 歳以上に限定、
                  (ii) 6 か月アウトカム窓の終端時点で 75 歳以上となる person-month を除外。
                  推計器として使う場合はこちらを推奨（有効年齢域 40–74 歳）。

推計時の扱い（論文 Fig. 2 / Fig. 4 と同一）
    - 指定しない共変量は学習標本の中央値（連続）/ 最頻値（カテゴリ）に固定
    - 点推定は log スケールの予測値を exp(·) − 1 で逆変換した値（= 論文の 17,170 円 等と同じ量。
      log 変換モデルの逆変換なので「幾何平均型」であり、算術平均（予算規模）より小さい）
    - 95%CI: 'model' = pyGAM の point-wise CI（log スケール）を逆変換（論文 Fig. 2 と同じ）
             'bootstrap' = 個人単位クラスターブートストラップ（論文 R1-7 と同じ再標本化）の
                           パーセンタイル CI（fit 時に --bootstrap N を指定した場合のみ）

使い方
    # 学習（1 回だけ）
    python steps_cost_estimator.py fit --data <CSV> --out steps_cost_gam.pkl \
        [--bootstrap 200 --n-jobs 2] [--lookup lookup_table.csv]

    # 推計（単発）
    python steps_cost_estimator.py predict --model steps_cost_gam.pkl --steps 8000 --age 68
    python steps_cost_estimator.py predict --model steps_cost_gam.pkl --steps 8000 --age 68 --variant paper

    # 推計（CSV 一括: steps, age [, sex, bmi, health_check, past_cost_6m, covid] 列）
    python steps_cost_estimator.py predict --model steps_cost_gam.pkl --csv input.csv --out pred.csv

    # Python から
    from steps_cost_estimator import StepsCostEstimator
    est = StepsCostEstimator.load('steps_cost_gam.pkl')
    est.predict(steps=8000, age=68)                               # corrected, model CI
    est.predict(steps=[5000, 11000], age=[55, 72], ci_method='bootstrap')
    est.predict(steps=10893, age=68, variant='paper')             # 論文値の再現
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
    raise ImportError("pygam が必要です: pip install pygam") from e


# ==========================================================================
# 仕様（論文と同一。変更しないことを推奨）
# ==========================================================================
LAG = 6                                   # 主要 horizon（a priori）
N_SPLINES_STEPS = 10                      # Steps の平滑化基底数
ANALYSIS_START = "2020-06-01"
ANALYSIS_END = "2023-06-01"
STEPS_UPPER_PCTL = 0.995                  # 予測の歩数上限（99.5 %ile）
DENSE_PCTL = (0.025, 0.975)               # データ密な範囲（解釈可能域）
BMI_IMPUTE_DECIMALS = 1                   # 個人別平均 BMI の中央値（22.7）で補完
AGE_REFERENCE_YEAR = 2023                 # データの Age が定義された年
ELDERLY_SWITCH_AGE = 75                   # 国保 → 後期高齢者医療制度 への移行年齢

FEATURE_NAMES = ["Steps", "Age", "Sex", "BMI", "HealthCheck", "PastCost_6m_Sum", "COVID19"]
FACTOR_NAMES = {"Sex", "HealthCheck", "COVID19"}
TARGET_COL = f"FutureCost_{LAG}m_Sum"

VARIANTS = {
    "paper": dict(label="論文と同一の標本（公表値を再現）", age_mode="reference",
                  exclude_post_switch=False, min_age=None, valid_age_max=74),
    "corrected": dict(label="75歳到達後の月を除外・年齢を各月時点に換算（推計用, 既定）",
                      age_mode="index_month", exclude_post_switch=True, min_age=40, valid_age_max=None),
}
DEFAULT_VARIANT = "corrected"

# 入力名 → 特徴量名（predict の引数 / CSV 列名）
INPUT_ALIASES = {
    "steps": "Steps", "age": "Age", "sex": "Sex", "bmi": "BMI",
    "health_check": "HealthCheck", "past_cost_6m": "PastCost_6m_Sum", "covid": "COVID19",
}


def _build_terms():
    # 列順は FEATURE_NAMES に固定: Steps, Age, Sex, BMI, HealthCheck, PastCost, COVID19
    return (s(0, n_splines=N_SPLINES_STEPS) + s(1) + f(2) + s(3) + f(4) + s(5) + f(6))


# ==========================================================================
# 前処理（論文コード load_and_engineer / slice_analysis と同一ロジック）
# ==========================================================================
def load_and_engineer(path_or_df, bmi_col: str = "BMI 2") -> tuple[pd.DataFrame, float]:
    """
    CSV を読み、PastCost_6m_Sum / FutureCost_6m_Sum を生成し、BMI を中央値補完する。
    bmi_col: 欠損が NaN で入っている BMI 列（本データでは 'BMI 2'。'BMI' は 0 = 欠損の旧コード）。
    返り値: (df, bmi_impute_value)
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

    # BMI: 個人別平均の中央値（論文: 22.7）で補完
    bmi_impute = round(float(df.groupby("uuid")[bmi_col].mean().median()), BMI_IMPUTE_DECIMALS)
    df["BMI_raw"] = df[bmi_col]
    df["BMI"] = df[bmi_col].fillna(bmi_impute)

    # 過去 6 か月累積（baseline）/ 将来 6 か月累積（index 月を除く）
    df["PastCost_6m_Sum"] = df.groupby("uuid")["Cost"].transform(
        lambda x: x.shift(1).rolling(window=6, min_periods=6).sum())
    indexer = pd.api.indexers.FixedForwardWindowIndexer(window_size=LAG)
    df[TARGET_COL] = df.groupby("uuid")["Cost"].transform(
        lambda x: x.shift(-1).rolling(window=indexer).sum())
    return df, bmi_impute


def slice_analysis(df: pd.DataFrame, start=ANALYSIS_START, end=ANALYSIS_END) -> pd.DataFrame:
    """分析期間で切り出し、6 か月アウトカムと baseline が揃った行を抽出（論文と同一）"""
    d = df[(df["Date"] >= pd.Timestamp(start)) & (df["Date"] <= pd.Timestamp(end))].copy()
    d = d.dropna(subset=[TARGET_COL, "PastCost_6m_Sum", "Steps"] + FEATURE_NAMES).copy()
    return d.reset_index(drop=True)


def apply_variant(d: pd.DataFrame, variant: str) -> pd.DataFrame:
    """バリアントごとの標本定義を適用。'Age' 列を学習に使う年齢に置き換えて返す。"""
    v = VARIANTS[variant]
    d = d.copy()
    d["Age_ref"] = d["Age"]                                   # 2023 年時点の年齢（元データ）
    year_t = d["Date"].dt.year
    d["Age_t"] = d["Age_ref"] - (AGE_REFERENCE_YEAR - year_t)  # index 月時点の年齢（年単位）
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
# 推計器
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
    steps_upper: float = np.nan            # 99.5 %ile（予測上限）
    steps_dense: tuple = (np.nan, np.nan)  # 2.5–97.5 %ile（解釈可能域）
    age_range: tuple = (np.nan, np.nan)    # 学習標本の年齢範囲
    age_valid: tuple = (np.nan, np.nan)    # 推計に使ってよい年齢範囲
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
    """歩数 × 年齢 → 6 か月総医療費（点推定 + 95%CI）"""

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
                      f"{d['uuid'].nunique():,} individuals, age {d['Age'].min():.0f}–{d['Age'].max():.0f}")
            sub = self._fit_one(d, variant, bmi_impute)
            self.models[variant] = sub
            if verbose:
                i = sub.info
                print(f"[fit:{variant}] pseudo R² = {i.pseudo_r2:.4f}, AIC = {i.aic:,.0f}, "
                      f"EDoF = {i.edof_total:.1f}  ({time.time()-t0:.0f}s)")
                if i.local_nadir:
                    ln = i.local_nadir
                    print(f"[fit:{variant}] local nadir: {ln['steps']:,.0f} steps → {ln['cost']:,.0f} JPY "
                          f"[{ln['lower']:,.0f}–{ln['upper']:,.0f}] (age fixed at {i.fixed_covariates['Age']:.0f})")
                else:
                    print(f"[fit:{variant}] no strict local minimum inside the plotted range")
                gm = i.global_min
                print(f"[fit:{variant}] global min: {gm['steps']:,.0f} steps → {gm['cost']:,.0f} JPY")
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
        """個人（uuid）単位のクラスターブートストラップ（論文 R1-7 と同一の再標本化）"""
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
            for j, k in n_levels.items():            # factor 水準が欠ける稀な再標本は捨てる
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
            raise RuntimeError("未学習です。fit() か load() を先に実行してください")
        variant = variant or DEFAULT_VARIANT
        if variant not in self.models:
            raise ValueError(f"variant '{variant}' は学習されていません（利用可能: {list(self.models)}）")
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
            raise ValueError("steps と age の長さが一致しません")

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
            raise ValueError(f"学習範囲外の入力があります（steps {lo_s:.0f}–{hi_s:.0f}, age {lo_a:.0f}–{hi_a:.0f}）")
        if extrapolation == "clip":
            cols["Steps"] = np.clip(cols["Steps"], lo_s, hi_s)
            cols["Age"] = np.clip(cols["Age"], lo_a, hi_a)
        elif extrapolation != "allow":
            raise ValueError("extrapolation は 'clip' / 'allow' / 'raise'")

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
        steps : 月平均の日歩数（スカラー or 配列）
        age   : 年齢（スカラー or 配列）
        sex, bmi, health_check, past_cost_6m, covid :
            省略時は学習標本の中央値/最頻値に固定（論文 Fig.2 と同じ）。
            sex: 1=男性, 2=女性 / health_check: 0/1 / past_cost_6m: 過去6か月医療費[円] / covid: 0/1
        ci        : 信頼水準（既定 0.95）
        ci_method : 'model'（pyGAM の point-wise CI, 論文 Fig.2 と同じ）
                    'bootstrap'（個人単位クラスターブートストラップのパーセンタイル CI）
        variant   : 'corrected'（既定）/ 'paper'
        extrapolation : 'clip'（範囲外は端に丸める・既定）/ 'allow' / 'raise'

        Returns
        -------
        DataFrame: steps_input, age_input, steps_used, age_used, point, lower, upper,
                   in_dense_range, clipped, variant, ci_method, ci
        単位は円（6 か月累積）。point は log モデルの逆変換値（幾何平均型）。
        """
        if not (0 < ci < 1):
            raise ValueError("ci は (0,1) の値")
        sub = self._get(variant)
        X, meta = self._design(sub, steps, age, sex, bmi, health_check, past_cost_6m, covid, extrapolation)

        pred_log = sub.gam.predict(X)
        if ci_method == "model":
            band = sub.gam.confidence_intervals(X, width=ci)
            lo_log, hi_log = band[:, 0], band[:, 1]
        elif ci_method == "bootstrap":
            if not sub.boot_models:
                raise RuntimeError(f"variant '{sub.info.variant}' にはブートストラップ CI がありません"
                                   "（fit 時に --bootstrap N を指定してください）")
            P = np.vstack([m.predict(X) for m in sub.boot_models])        # (B, n)
            a = (1 - ci) / 2
            lo_log, hi_log = np.percentile(P, 100 * a, axis=0), np.percentile(P, 100 * (1 - a), axis=0)
        else:
            raise ValueError("ci_method は 'model' / 'bootstrap'")

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
        """steps × age の格子で予測（ルックアップ表の作成用）"""
        info = self._get(variant).info
        sg = np.arange(0, info.steps_upper + 1e-9, steps_step)
        ag = np.arange(np.ceil(info.age_valid[0]), np.floor(info.age_valid[1]) + 1e-9, age_step)
        S, A = np.meshgrid(sg, ag, indexing="ij")
        return self.predict(S.ravel(), A.ravel(), variant=variant, **kw)

    # ---------------------------------------------------------- diagnostics
    def _find_optima(self, sub: _SubModel, n_grid: int = 200, order: int = 5) -> tuple[dict, dict]:
        """論文と同じ格子（0–99.5%ile, 200 点）・他共変量固定で局所最小・全体最小を検出"""
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
# 曲面 JSON（data/surface_paper.json）からの推計 — pickle 不要
# ==========================================================================
def predict_from_lookup(surface_json, steps, age) -> pd.DataFrame:
    """
    公開リポジトリ同梱の曲面 JSON（data/surface_paper.json: 歩数 100 刻み × 年齢 1 歳刻み）から、
    log スケールの双線形補間で点推定と 95%CI を返す。学習済みモデルや pygam は不要。
    範囲外（歩数 0–22,119, 年齢 40–74）は端に丸める。
    """
    import json
    with open(surface_json, encoding="utf-8") as fh:
        S = json.load(fh)
    sg = np.asarray(S["steps"], float); ag = np.asarray(S["ages"], float)
    piv = {k: np.asarray(S[k], float) for k in ("point", "lower", "upper")}      # [age][step]
    dense_lo, dense_hi = S["meta"]["steps_dense"]
    steps = np.atleast_1d(np.asarray(steps, float)); age = np.atleast_1d(np.asarray(age, float))
    n = max(len(steps), len(age))
    steps = np.repeat(steps, n) if len(steps) == 1 else steps
    age = np.repeat(age, n) if len(age) == 1 else age
    s = np.clip(steps, sg[0], sg[-1]); a = np.clip(age, ag[0], ag[-1])
    i = np.clip(np.searchsorted(sg, s, side="right") - 1, 0, len(sg) - 2)
    j = np.clip(np.searchsorted(ag, a, side="right") - 1, 0, len(ag) - 2)
    ts = (s - sg[i]) / (sg[i + 1] - sg[i]); ta = (a - ag[j]) / (ag[j + 1] - ag[j])
    out = pd.DataFrame({"steps_input": steps, "age_input": age, "steps_used": s, "age_used": a,
                        "in_dense_range": (s >= dense_lo) & (s <= dense_hi),
                        "clipped": (steps != s) | (age != a)})
    for k, z in piv.items():
        z = np.log1p(z)
        v = (1 - ta) * ((1 - ts) * z[j, i] + ts * z[j, i + 1]) + ta * ((1 - ts) * z[j + 1, i] + ts * z[j + 1, i + 1])
        out[k] = np.round(np.expm1(v), 0)
    out["variant"] = "paper"; out["ci_method"] = "model"; out["ci"] = 0.95
    return out


# ==========================================================================
# CLI
# ==========================================================================
def _cli():
    ap = argparse.ArgumentParser(description="歩数×年齢 → 6か月総医療費 推計器")
    sub = ap.add_subparsers(dest="cmd", required=True)

    pf = sub.add_parser("fit", help="学習")
    pf.add_argument("--data", required=True)
    pf.add_argument("--out", default="steps_cost_gam.pkl")
    pf.add_argument("--bmi-col", default="BMI 2")
    pf.add_argument("--variants", nargs="+", default=["paper", "corrected"], choices=list(VARIANTS))
    pf.add_argument("--bootstrap", type=int, default=0, help="クラスターブートストラップ反復数（0 で省略）")
    pf.add_argument("--bootstrap-variants", nargs="+", default=[DEFAULT_VARIANT], choices=list(VARIANTS))
    pf.add_argument("--seed", type=int, default=42)
    pf.add_argument("--n-jobs", type=int, default=1)
    pf.add_argument("--lookup", default=None, help="steps×age ルックアップ表 CSV の出力先（既定バリアント）")

    pp = sub.add_parser("predict", help="推計")
    pp.add_argument("--model", default="steps_cost_gam.pkl")
    pp.add_argument("--lookup", default=None, help="曲面 JSON（data/surface_paper.json）から推計（pickle 不要, paper バリアント固定）")
    pp.add_argument("--steps", type=float)
    pp.add_argument("--age", type=float)
    pp.add_argument("--sex", type=float); pp.add_argument("--bmi", type=float)
    pp.add_argument("--health-check", type=float); pp.add_argument("--past-cost-6m", type=float)
    pp.add_argument("--covid", type=float)
    pp.add_argument("--csv", help="一括推計用 CSV（steps, age [, sex, bmi, health_check, past_cost_6m, covid]）")
    pp.add_argument("--out", help="一括推計の出力 CSV")
    pp.add_argument("--ci", type=float, default=0.95)
    pp.add_argument("--ci-method", choices=["model", "bootstrap"], default="model")
    pp.add_argument("--variant", choices=list(VARIANTS), default=DEFAULT_VARIANT)
    pp.add_argument("--extrapolation", choices=["clip", "allow", "raise"], default="clip")

    pi = sub.add_parser("info", help="学習済みモデルの要約")
    pi.add_argument("--model", default="steps_cost_gam.pkl")

    a = ap.parse_args()
    if a.cmd == "fit":
        est = StepsCostEstimator().fit(a.data, variants=a.variants, bmi_col=a.bmi_col,
                                       n_bootstrap=a.bootstrap, bootstrap_variants=a.bootstrap_variants,
                                       seed=a.seed, n_jobs=a.n_jobs)
        est.save(a.out)
        print(f"\n[fit] saved → {a.out}")
        if a.lookup:
            g = est.predict_grid(variant=DEFAULT_VARIANT)
            if est.models[DEFAULT_VARIANT].boot_models:
                gb = est.predict_grid(variant=DEFAULT_VARIANT, ci_method="bootstrap")
                g["lower_boot"], g["upper_boot"] = gb["lower"], gb["upper"]
            g.to_csv(a.lookup, index=False)
            print(f"[fit] lookup table → {a.lookup} ({len(g):,} rows)")
    elif a.cmd == "predict" and a.lookup:
        if a.csv:
            df = pd.read_csv(a.csv)
            res = predict_from_lookup(a.lookup, df["steps"].values, df["age"].values)
            if a.out:
                res.to_csv(a.out, index=False); print(f"→ {a.out} ({len(res):,} rows)")
            else:
                print(res.to_string(index=False))
        else:
            if a.steps is None or a.age is None:
                ap.error("--steps と --age（または --csv）を指定してください")
            r = predict_from_lookup(a.lookup, a.steps, a.age).iloc[0]
            print(f"steps={r.steps_used:,.0f}/day, age={r.age_used:.0f} → 6か月医療費 {r.point:,.0f} 円  "
                  f"95%CI [{r.lower:,.0f} – {r.upper:,.0f}]  (lookup, paper variant)")
    elif a.cmd == "predict":
        est = StepsCostEstimator.load(a.model)
        kw = dict(ci=a.ci, ci_method=a.ci_method, variant=a.variant, extrapolation=a.extrapolation)
        if a.csv:
            df = pd.read_csv(a.csv)
            cols = {k: df[k].values for k in INPUT_ALIASES if k in df.columns}
            if "steps" not in cols or "age" not in cols:
                ap.error("CSV に steps, age 列が必要です")
            res = est.predict(**cols, **kw)
            if a.out:
                res.to_csv(a.out, index=False); print(f"→ {a.out} ({len(res):,} rows)")
            else:
                print(res.to_string(index=False))
        else:
            if a.steps is None or a.age is None:
                ap.error("--steps と --age（または --csv）を指定してください")
            res = est.predict(a.steps, a.age, sex=a.sex, bmi=a.bmi, health_check=a.health_check,
                              past_cost_6m=a.past_cost_6m, covid=a.covid, **kw)
            r = res.iloc[0]
            info = est.models[a.variant].info
            print(f"steps={r.steps_used:,.0f}/day, age={r.age_used:.0f} → "
                  f"6か月医療費 {r.point:,.0f} 円  {int(a.ci*100)}%CI [{r.lower:,.0f} – {r.upper:,.0f}]"
                  f"  (variant={a.variant}, CI={a.ci_method})")
            if r.clipped:
                print(f"  ※ 入力が有効範囲外（steps 0–{info.steps_upper:,.0f}, age "
                      f"{info.age_valid[0]:.0f}–{info.age_valid[1]:.0f}）のため端に丸めました")
            if not r.in_dense_range:
                print(f"  ※ 歩数がデータ密な範囲（{info.steps_dense[0]:,.0f}–{info.steps_dense[1]:,.0f}）外: 外挿として解釈")
    elif a.cmd == "info":
        est = StepsCostEstimator.load(a.model)
        print(json.dumps(est.summary(), ensure_ascii=False, indent=2, default=float))


if __name__ == "__main__":
    _cli()
