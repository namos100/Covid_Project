"""
Basic vs. Improved model for the wave-2 census -> covid-outcome question.

Basic ML Model : plain multiple linear regression (OLS), all 19 usable
                  census features together, per target.
Improved Model  : XGBoost regression, same features/targets, same
                  cross-validation scheme - built specifically to see
                  whether a non-linear model captures signal the linear
                  model misses.

Both models are evaluated identically so the comparison is fair:
  - Same 19 census features (see "feature selection" section below for why
    4 raw columns were dropped entirely).
  - Same rows (816 of 1008 areas - the ones with no missing values across
    the 19 features).
  - Same 5-fold GROUPED cross-validation, grouped by town (town_code
    extracted from City_agas_code) - NOT a plain shuffled KFold. Sub-areas
    of the same city are highly correlated with each other; a plain KFold
    could put two sub-areas of Jerusalem in different folds and let the
    model "leak" city-level information between train and test, inflating
    CV R^2. GroupKFold keeps every sub-area of a city on the same side of
    the split, matching the "geographic leakage" item on the project's own
    leakage checklist.

Feature selection: 4 of the 23 continuous census columns used in the
original (pre-rebuild) build_models.ipynb are dropped here -
pop_density, inst_pcnt, Foreign_pcnt, change_pcnt. Checked directly against
Mifkad_2_processed.csv: these are effectively never populated at sub-area
resolution (e.g. inst_pcnt: 233/3857 sub-area rows, 0/145 whole-city rows) -
this is real sparsity in the census source, not a processing bug. Keeping
them would have dropped the model table from 1008 to 31 usable rows.

Target preprocessing: slope_to_peak_IR is winsorized at the 1st/99th
percentile before modeling (see EDA: skew=15.35 on the unfiltered table -
a couple of areas whose "peak" landed 1-2 days after the window start blow
up the slope estimate and would otherwise dominate the loss for both
models).
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

TABLE_PATH = Path("/home/bcrlab/igguest/porat_naama/data/processed/models/wave2_model_table.csv")
OUT_DIR = Path("/home/bcrlab/igguest/porat_naama/data/processed/models")
FIG_DIR = OUT_DIR / "figures_wave2"
FIG_DIR.mkdir(parents=True, exist_ok=True)

TARGETS = ["mean_IR", "peak_IR", "slope_to_peak_IR", "severity_hosp_per_case"]

SPARSE_DROPPED = ["pop_density", "inst_pcnt", "Foreign_pcnt", "change_pcnt"]
FEATURE_COLS = [
    "pop_approx", "hh_total_approx", "sexRatio", "WrkOutLoc_pcnt", "AcadmCert_pcnt",
    "age65_pcnt", "j_isr_pcnt", "size_avg", "WrkY_pcnt", "employeesAnnual_medWage",
    "Vehicle2up_pcnt", "koshi5_pcnt",
    "age_structure_idx", "origin_idx", "household_family_idx", "employment_idx",
    "income_idx", "housing_assets_idx", "disability_idx",
]

N_SPLITS = 5
RANDOM_STATE = 0

# ============================================================
# Load + prepare
# ============================================================
df = pd.read_csv(TABLE_PATH)
print(f"Loaded {len(df):,} areas, {df.shape[1]} columns")

df["town_code"] = df["City_agas_code"].str.split("_").str[0]

n_before = len(df)
df = df.dropna(subset=FEATURE_COLS).reset_index(drop=True)
print(f"Dropped {SPARSE_DROPPED} (near-total missingness at sub-area level - see docstring)")
print(f"Rows with all {len(FEATURE_COLS)} remaining features present: {len(df):,} / {n_before:,}")

# Winsorize the one extremely skewed target
lo, hi = df["slope_to_peak_IR"].quantile([0.01, 0.99])
n_clipped = ((df["slope_to_peak_IR"] < lo) | (df["slope_to_peak_IR"] > hi)).sum()
df["slope_to_peak_IR"] = df["slope_to_peak_IR"].clip(lo, hi)
print(f"Winsorized slope_to_peak_IR at [{lo:.4f}, {hi:.4f}] - {n_clipped} values clipped")

groups = df["town_code"]
n_groups = groups.nunique()
print(f"Grouped CV: {n_groups} towns across {len(df)} areas\n")

X_raw = df[FEATURE_COLS].values
scaler = StandardScaler()
X = scaler.fit_transform(X_raw)

# ============================================================
# Fit both models on every target, with the same grouped CV
# ============================================================
results = []
predictions = {}  # (target, model_name) -> (y_true, y_pred_cv)
importances = {}  # target -> xgboost feature importances

gkf = GroupKFold(n_splits=N_SPLITS)

models = {
    "Linear (Basic)": LinearRegression(),
    "XGBoost (Improved)": XGBRegressor(
        n_estimators=200, max_depth=3, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, random_state=RANDOM_STATE,
        n_jobs=4,  # shared machine (64 cores) - avoid thread-count explosion across 20 fits
    ),
}

for target in TARGETS:
    y = df[target].values
    for model_name, model in models.items():
        y_pred_cv = cross_val_predict(model, X, y, cv=gkf, groups=groups)
        cv_r2 = r2_score(y, y_pred_cv)
        cv_mae = mean_absolute_error(y, y_pred_cv)

        model.fit(X, y)
        y_pred_in = model.predict(X)
        in_sample_r2 = r2_score(y, y_pred_in)

        results.append({
            "target": target,
            "model": model_name,
            "cv_r2": cv_r2,
            "cv_mae": cv_mae,
            "in_sample_r2": in_sample_r2,
            "overfit_gap": in_sample_r2 - cv_r2,
        })
        predictions[(target, model_name)] = (y, y_pred_cv)

        if model_name == "XGBoost (Improved)":
            importances[target] = pd.Series(model.feature_importances_, index=FEATURE_COLS).sort_values(ascending=False)

        print(f"{target:28s} | {model_name:20s} | CV R2={cv_r2:+.3f}  CV MAE={cv_mae:.4f}  in-sample R2={in_sample_r2:+.3f}")
    print()

results_df = pd.DataFrame(results)
results_path = OUT_DIR / "wave2_basic_vs_improved_results.csv"
results_df.to_csv(results_path, index=False, encoding="utf-8-sig")
print("Saved:", results_path)

# ============================================================
# Figure 1: actual vs. CV-predicted, Linear vs XGBoost, one row per target
# ============================================================
fig, axes = plt.subplots(len(TARGETS), 2, figsize=(9, 4 * len(TARGETS)))
for row, target in enumerate(TARGETS):
    for col, model_name in enumerate(models.keys()):
        ax = axes[row, col]
        y_true, y_pred = predictions[(target, model_name)]
        r2 = results_df.query("target == @target and model == @model_name")["cv_r2"].iloc[0]

        ax.scatter(y_true, y_pred, s=12, alpha=0.4, color="#2E6F95" if col == 0 else "#C1440E")
        lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
        ax.plot(lims, lims, "k--", linewidth=1, alpha=0.6)
        ax.set_title(f"{target}\n{model_name}  (CV R²={r2:.3f})", fontsize=10)
        ax.set_xlabel("actual")
        ax.set_ylabel("predicted (out-of-fold)")

plt.tight_layout()
fig.savefig(FIG_DIR / "actual_vs_predicted_basic_vs_improved.png", dpi=150)
plt.close(fig)
print("Saved:", FIG_DIR / "actual_vs_predicted_basic_vs_improved.png")

# ============================================================
# Figure 2: summary bar chart, CV R^2 per target x model
# ============================================================
pivot = results_df.pivot(index="target", columns="model", values="cv_r2").reindex(TARGETS)
fig, ax = plt.subplots(figsize=(8, 5))
x = np.arange(len(TARGETS))
width = 0.35
ax.bar(x - width / 2, pivot["Linear (Basic)"], width, label="Linear (Basic)", color="#2E6F95")
ax.bar(x + width / 2, pivot["XGBoost (Improved)"], width, label="XGBoost (Improved)", color="#C1440E")
ax.axhline(0, color="black", linewidth=0.8)
ax.set_xticks(x)
ax.set_xticklabels(TARGETS, rotation=20, ha="right")
ax.set_ylabel("5-fold grouped CV R² (out-of-sample)")
ax.set_title("Basic vs. Improved model: out-of-sample R² per target")
ax.legend()
plt.tight_layout()
fig.savefig(FIG_DIR / "cv_r2_summary_basic_vs_improved.png", dpi=150)
plt.close(fig)
print("Saved:", FIG_DIR / "cv_r2_summary_basic_vs_improved.png")

# ============================================================
# Figure 3: overfit gap (in-sample R^2 - CV R^2) per target x model
# ============================================================
pivot_gap = results_df.pivot(index="target", columns="model", values="overfit_gap").reindex(TARGETS)
fig, ax = plt.subplots(figsize=(8, 5))
ax.bar(x - width / 2, pivot_gap["Linear (Basic)"], width, label="Linear (Basic)", color="#2E6F95")
ax.bar(x + width / 2, pivot_gap["XGBoost (Improved)"], width, label="XGBoost (Improved)", color="#C1440E")
ax.axhline(0, color="black", linewidth=0.8)
ax.set_xticks(x)
ax.set_xticklabels(TARGETS, rotation=20, ha="right")
ax.set_ylabel("in-sample R²  −  CV R²  (overfitting gap)")
ax.set_title("Overfitting: how much each model 'cheats' on training data vs. held-out folds")
ax.legend()
plt.tight_layout()
fig.savefig(FIG_DIR / "overfit_gap_basic_vs_improved.png", dpi=150)
plt.close(fig)
print("Saved:", FIG_DIR / "overfit_gap_basic_vs_improved.png")

# ============================================================
# Figure 4: XGBoost feature importance, one panel per target
# ============================================================
fig, axes = plt.subplots(1, len(TARGETS), figsize=(5 * len(TARGETS), 5))
for ax, target in zip(axes, TARGETS):
    imp = importances[target].head(8).iloc[::-1]
    ax.barh(imp.index, imp.values, color="#C1440E")
    ax.set_title(target, fontsize=10)
    ax.set_xlabel("XGBoost importance")
plt.tight_layout()
fig.savefig(FIG_DIR / "xgboost_feature_importance.png", dpi=150)
plt.close(fig)
print("Saved:", FIG_DIR / "xgboost_feature_importance.png")

print("\n=== SUMMARY TABLE (CV R2) ===")
print(pivot.round(3))
