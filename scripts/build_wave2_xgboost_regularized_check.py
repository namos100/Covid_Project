"""
Follow-up check: does a much more heavily regularized XGBoost close the gap
with the Linear (Basic) model from build_wave2_basic_and_improved_models.py?

Same data prep, same grouped 5-fold CV (grouped by town, to avoid geographic
leakage between sub-areas of the same city). Only the XGBoost hyperparameters
change: shallower trees, fewer estimators, higher min_child_weight, explicit
L1/L2 regularization - all aimed at reducing the severe overfitting seen in
the first run (in-sample R2 up to 0.87 vs CV R2 as low as -0.11).
"""

from pathlib import Path

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

TABLE_PATH = Path("/home/bcrlab/igguest/porat_naama/data/processed/models/wave2_model_table.csv")
OUT_DIR = Path("/home/bcrlab/igguest/porat_naama/data/processed/models")

TARGETS = ["mean_IR", "peak_IR", "slope_to_peak_IR", "severity_hosp_per_case"]
FEATURE_COLS = [
    "pop_approx", "hh_total_approx", "sexRatio", "WrkOutLoc_pcnt", "AcadmCert_pcnt",
    "age65_pcnt", "j_isr_pcnt", "size_avg", "WrkY_pcnt", "employeesAnnual_medWage",
    "Vehicle2up_pcnt", "koshi5_pcnt",
    "age_structure_idx", "origin_idx", "household_family_idx", "employment_idx",
    "income_idx", "housing_assets_idx", "disability_idx",
]
N_SPLITS = 5
RANDOM_STATE = 0

df = pd.read_csv(TABLE_PATH)
df["town_code"] = df["City_agas_code"].str.split("_").str[0]
df = df.dropna(subset=FEATURE_COLS).reset_index(drop=True)

lo, hi = df["slope_to_peak_IR"].quantile([0.01, 0.99])
df["slope_to_peak_IR"] = df["slope_to_peak_IR"].clip(lo, hi)

groups = df["town_code"]
X = StandardScaler().fit_transform(df[FEATURE_COLS].values)
gkf = GroupKFold(n_splits=N_SPLITS)

models = {
    "Linear (Basic)": LinearRegression(),
    "XGBoost (original, from before)": XGBRegressor(
        n_estimators=200, max_depth=3, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, random_state=RANDOM_STATE, n_jobs=4,
    ),
    "XGBoost (regularized)": XGBRegressor(
        n_estimators=40, max_depth=2, learning_rate=0.05,
        min_child_weight=8, subsample=0.6, colsample_bytree=0.6,
        reg_alpha=1.0, reg_lambda=5.0,
        random_state=RANDOM_STATE, n_jobs=4,
    ),
}

rows = []
for target in TARGETS:
    y = df[target].values
    for name, model in models.items():
        y_pred_cv = cross_val_predict(model, X, y, cv=gkf, groups=groups)
        cv_r2 = r2_score(y, y_pred_cv)
        cv_mae = mean_absolute_error(y, y_pred_cv)
        model.fit(X, y)
        in_r2 = r2_score(y, model.predict(X))
        rows.append({"target": target, "model": name, "cv_r2": cv_r2, "cv_mae": cv_mae,
                      "in_sample_r2": in_r2, "overfit_gap": in_r2 - cv_r2})
        print(f"{target:26s} | {name:34s} | CV R2={cv_r2:+.3f}  in-sample R2={in_r2:+.3f}  gap={in_r2 - cv_r2:+.3f}")
    print()

results = pd.DataFrame(rows)
out_path = OUT_DIR / "wave2_xgboost_regularized_check.csv"
results.to_csv(out_path, index=False, encoding="utf-8-sig")
print("Saved:", out_path)

print("\n=== CV R2 pivot ===")
print(results.pivot(index="target", columns="model", values="cv_r2").reindex(TARGETS).round(3))
