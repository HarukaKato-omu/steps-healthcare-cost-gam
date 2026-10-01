# Daily steps and 6-month healthcare expenditure — estimator from Kato (2026)

[日本語は下にあります](#日本語)

Point estimate and 95% CI of 6-month healthcare expenditure from **daily steps** and **age**, using the
Generalized Additive Model published in:

> Kato, H. (2026). Non-linear association between mHealth-measured daily steps and subsequent healthcare
> expenditures among long-term app users in Osaka, Japan. *SSM – Population Health*, 35, 101958.
> https://doi.org/10.1016/j.ssmph.2026.101958

**Web tool (GitHub Pages):** `docs/index.html` — enter steps and age, read the estimate, and see the point on the
fitted steps × age surface (the paper's Fig. 4).

## What is public and what is not

| Public (this repository) | Not public |
|---|---|
| Model specification and training code (`src/`), page build script (`scripts/`) | Individual-level data (A-Smile step counts linked to insurance claims) |
| The fitted surface on a 100-step × 1-year grid (`data/surface_paper.json`) — a derived aggregate of the model, nothing else | The fitted model file |
| The static page (`docs/`) | |

The training script runs only in the author's environment. The page and the Python lookup run from the JSON.

## Read this before using the numbers

- **An association, not an intervention effect.** The curve describes expenditure differences across people with
  different step counts in one selected cohort (long-term A-Smile users aged 40+ in Osaka, linked to National Health
  Insurance claims, 2020–2023), adjusted for age, sex, BMI, health-check attendance, baseline expenditure and
  emergency periods. It does not estimate what would happen if a person changed their steps.
- **A geometric-mean-type quantity.** The model is fitted on log(Y+1); the back-transformed prediction is far below the
  arithmetic mean of 6-month expenditures. Do not sum the estimates into budget totals.
- **Point-wise 95% CI of the fitted curve** (as in the paper's Fig. 2), not a prediction interval for an individual.
- **Ages 40–74.** Covariates other than steps and age are held at the sample median/mode (female, BMI 22.7, no health
  check, baseline 6-month expenditure 75,220 JPY, non-emergency period), as in the paper's figures. Steps outside the
  2.5–97.5th percentile of observations (845–14,906/day) are flagged as extrapolated.

## Model

```
log(Y6 + 1) = α + s1(Steps; k = 10) + s2(Age) + s3(BMI) + s4(Baseline6) + β·[Sex, HealthCheck, COVID-19]
```
Gaussian GAM, identity link (pyGAM 0.12.0), penalties at pyGAM defaults; outcome = expenditure over the 6 months after
the index month, baseline = the 6 months before. 338,706 person-months from 19,599 individuals (June 2020 – June 2023);
explained deviance 0.326. The refit reproduces the published sample size, explained deviance and the positions of the
local nadir (10,893 steps/day) and global minimum (19,674 steps/day).

## Use

**Web:** open `docs/index.html` (or the GitHub Pages site). Between grid points, values are interpolated on the log
scale; at grid points they equal the model output to the yen.

**Python (no pygam needed):**
```bash
pip install numpy pandas
python src/steps_cost_estimator.py predict --lookup data/surface_paper.json --steps 8000 --age 68
#  steps=8,000/day, age=68 → 6か月医療費 19,044 円  95%CI [18,278 – 19,843]
python src/steps_cost_estimator.py predict --lookup data/surface_paper.json --csv inputs.csv --out out.csv   # columns: steps, age
```
```python
from steps_cost_estimator import predict_from_lookup
predict_from_lookup("data/surface_paper.json", steps=[5000, 11000], age=[55, 72])
```

**Retraining and page build (author only, data required):**
```bash
pip install -r requirements.txt
python src/steps_cost_estimator.py fit --data <linked_csv> --out steps_cost_gam.pkl
python scripts/build_site.py --model steps_cost_gam.pkl --repo-url <this repository>
```

## Citation and license

Please cite the paper above; `CITATION.cff` describes this software. Code and page: MIT. `data/surface_paper.json`: CC BY 4.0.

---

## 日本語

Kato (2026, *SSM – Population Health* 35:101958) の一般化加法モデル（GAM）を用いて、**日歩数（月平均）**と**年齢**から
**6か月医療費の点推定と95%信頼区間**を返すツールです。`docs/index.html`（GitHub Pages）では、入力値を論文 Fig. 4 の
歩数×年齢曲面の上に赤点で表示します。

**公開するもの**：モデル仕様と学習コード、ページ生成スクリプト、100歩×1歳格子上の推定曲面（モデルの派生集計のみ）、静的ページ。
**公開しないもの**：個人データ（アスマイル歩数と国保レセプトの連結データ）、学習済みモデルファイル。学習コードは著者環境でのみ実行します。

**使う前に**

- 曲線は選択されたコホート内の観察的な関連であり、歩数を変えたときの医療費変化（介入効果）の推計ではありません。
- 値は log(Y+1) モデルの逆変換値（幾何平均型）で、6か月医療費の算術平均より大幅に小さく、予算総額の算出には使えません。
- 95%CI は推定曲線の point-wise 区間（論文 Fig. 2 と同じ）で、個人の予測区間ではありません。
- 対象年齢は 40–74 歳。歩数・年齢以外の共変量は標本の中央値／最頻値に固定（女性、BMI 22.7、健診なし、過去6か月医療費
  75,220 円、非宣言期間）。観測の 2.5–97.5 パーセンタイル（845–14,906 歩）の外は外挿として表示します。

**Python から**（pygam 不要）：
```bash
python src/steps_cost_estimator.py predict --lookup data/surface_paper.json --steps 8000 --age 68
```
