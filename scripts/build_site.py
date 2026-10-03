# -*- coding: utf-8 -*-
"""
build_site.py - build the public artefacts from the fitted model (paper variant):
  data/fitted_surface_grid.json   fitted steps x age surface (point estimate, 95% CI) + metadata
  docs/index.html                 GitHub Pages page (docs/template.html with the grid embedded)

The person-month panel and the fitted model file are not part of the repository; this script runs in
the author's environment only.

Usage:  python scripts/build_site.py --model /path/to/steps_cost_gam.pkl [--steps-max 20000] [--repo-url URL]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from steps_cost_estimator import StepsCostEstimator  # noqa: E402

AGE_MIN, AGE_MAX = 40, 74        # age range of the public tool (claims end at 75: insurance switch)
STEPS_STEP = 100                 # grid spacing for steps
STEPS_MAX = 20000                # upper limit of the public tool (data-dense range ends at ~14,906)
VARIANT = "paper"                # the model fitted on the paper's own sample
GRID_JSON = ROOT / "data" / "fitted_surface_grid.json"
TEMPLATE = ROOT / "docs" / "template.html"
PAGE = ROOT / "docs" / "index.html"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="fitted model file written by steps_cost_estimator.py fit")
    ap.add_argument("--repo-url", default="https://github.com/HarukaKato-omu/steps-healthcare-cost-gam")
    ap.add_argument("--steps-max", type=float, default=STEPS_MAX)
    ap.add_argument("--fragment", default=None, help="also write the page without the HTML skeleton (preview use)")
    a = ap.parse_args()

    est = StepsCostEstimator.load(a.model)
    sub = est.models[VARIANT]
    info = sub.info
    upper = min(float(info.steps_upper), a.steps_max)

    steps = list(range(0, int(upper // STEPS_STEP) * STEPS_STEP + 1, STEPS_STEP))
    if steps[-1] < upper - 1:
        steps.append(round(upper))
    ages = list(range(AGE_MIN, AGE_MAX + 1))
    S, A = np.meshgrid(steps, ages, indexing="ij")          # S[i_step, j_age]
    res = est.predict(S.ravel(), A.ravel(), variant=VARIANT, ci_method="model", extrapolation="allow")

    def mat(col):
        return np.round(res[col].values.reshape(len(steps), len(ages))).astype(int).T.tolist()  # [age][step]

    meta = dict(
        model="LinearGAM: log(Y6+1) ~ s(Steps,k=10)+s(Age)+f(Sex)+s(BMI)+f(HealthCheck)+s(PastCost6)+f(COVID19)",
        variant=VARIANT,
        n_person_months=info.n_person_months, n_individuals=info.n_individuals,
        analysis_period=f"{info.analysis_start[:7]} to {info.analysis_end[:7]}",
        pseudo_r2=round(info.pseudo_r2, 4), aic=round(info.aic), edof=round(info.edof_total, 1),
        steps_upper=round(upper), steps_upper_paper=round(float(info.steps_upper)),
        steps_dense=[round(info.steps_dense[0]), round(info.steps_dense[1])],
        age_range=[AGE_MIN, AGE_MAX],
        fixed_covariates={k: (round(v, 1) if k == "BMI" else round(v)) for k, v in info.fixed_covariates.items()
                          if k not in ("Steps", "Age")},
        local_nadir={k: round(v) for k, v in info.local_nadir.items()},
        ci="95% point-wise (pyGAM confidence_intervals on log scale, back-transformed)",
        scale="geometric-mean type: exp(E[log(Y+1)]) - 1; not an arithmetic mean",
        software=est.meta.get("versions", {}),
        built=pd.Timestamp.now().strftime("%Y-%m-%d"),
        paper="Kato H. (2026) SSM - Population Health 35:101958. https://doi.org/10.1016/j.ssmph.2026.101958",
    )
    surface = dict(meta=meta, steps=steps, ages=ages, point=mat("point"), lower=mat("lower"), upper=mat("upper"))

    GRID_JSON.parent.mkdir(exist_ok=True)
    with open(GRID_JSON, "w", encoding="utf-8") as fh:
        json.dump(surface, fh, ensure_ascii=False, separators=(",", ":"))

    template = TEMPLATE.read_text(encoding="utf-8")
    payload = json.dumps(surface, ensure_ascii=False, separators=(",", ":"))
    assert "/*__SURFACE_JSON__*/" in template and "/*__REPO_URL__*/" in template
    fragment = template.replace("/*__SURFACE_JSON__*/", payload).replace("/*__REPO_URL__*/", a.repo_url)
    head, rest = fragment.split("<style>", 1)
    style, body = rest.split("</style>", 1)
    full = ("<!doctype html>\n<html lang=\"en\">\n<head>\n<meta charset=\"utf-8\">\n"
            "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1, viewport-fit=cover\">\n"
            "<meta name=\"description\" content=\"Point estimate and 95% CI of 6-month healthcare expenditure from daily steps and age, "
            "from the GAM in Kato (2026), SSM - Population Health.\">\n"
            + head + "<style>" + style + "</style>\n</head>\n<body>\n" + body + "\n</body>\n</html>\n")
    PAGE.write_text(full, encoding="utf-8")
    if a.fragment:
        Path(a.fragment).write_text(fragment, encoding="utf-8")
    print(f"grid: {len(steps)} steps x {len(ages)} ages -> {GRID_JSON.relative_to(ROOT)}, "
          f"{PAGE.relative_to(ROOT)} ({len(full)/1024:.0f} KB)")


if __name__ == "__main__":
    main()
