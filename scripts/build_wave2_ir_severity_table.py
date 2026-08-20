"""
Build the wave-2 IR/severity model table directly from the raw corona file.

Produces one row per City_agas_code for the window 2020-06-01 - 2020-10-31
(decline of wave 1 -> end of the second lockdown), with exactly 4 model
features:
  - mean_IR                : mean infection_rate over the window
  - peak_IR                : max infection_rate over the window
  - slope_to_peak_IR       : linear slope of infection_rate from the window
                              start to the day of the peak (speed of rise)
  - severity_hosp_per_case : sum(hospitalized_daily) / sum(cases_daily) over
                              the whole window (cumulative ratio, not daily,
                              to avoid small-count noise)

Only the raw source file is used - data/raw/geographic-sum-per-day-ver_00859.csv.
No census join here; none of these 4 features need a population denominator.

Documented assumptions (flagged explicitly, not silently applied):
  - '<15' censored values are replaced with 0 (matches what the rest of the
    processed pipeline actually does in preprocessing_corona_porat.ipynb,
    NOT the 7 documented as the project convention in COVID-Project-CLAUDE.md -
    this is a known open inconsistency in the project, resolved here as 0
    for consistency with the rest of the pipeline).
  - infection_rate uses the same adaptive-window method already established
    in preprocessing_corona_porat.ipynb (MIN_TESTS=30, up to 14 days back).
  - Negative daily diffs (data corrections in the accumulated counts) are
    clipped to 0.
  - Both city-aggregate rows (is_city_aggregate=1) and sub-area rows are
    included in the output, flagged via is_city_aggregate - filter to one
    consistent granularity before using this table in a regression against
    census features.
"""

from pathlib import Path

import numpy as np
import pandas as pd

RAW_PATH = Path("/home/bcrlab/igguest/porat_naama/data/raw/geographic-sum-per-day-ver_00859.csv")
OUT_DIR = Path("/home/bcrlab/igguest/porat_naama/data/processed/models")
OUT_DIR.mkdir(parents=True, exist_ok=True)

WINDOW_START = pd.Timestamp("2020-06-01")
WINDOW_END = pd.Timestamp("2020-10-31")

MISSING_VALUE_FILL = 0  # '<15' -> 0, see module docstring
MIN_TESTS_ADAPTIVE = 30
MAX_ADAPTIVE_WINDOW_DAYS = 14

MIN_TESTS_IN_WINDOW = 30  # minimum cumulative tests in the window for mean/peak IR to be trusted
MIN_CASES_IN_WINDOW = 30  # minimum cumulative cases in the window for the severity ratio to be trusted


# ============================================================
# Load raw data
# ============================================================
usecols = [
    "town_code", "agas_code", "town", "date",
    "accumulated_cases", "accumulated_diagnostic_tests", "accumulated_hospitalized",
]
df = pd.read_csv(RAW_PATH, usecols=usecols, encoding="utf-8-sig")

df = df.replace("<15", MISSING_VALUE_FILL)
for c in ["accumulated_cases", "accumulated_diagnostic_tests", "accumulated_hospitalized"]:
    df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)

df["town_code"] = pd.to_numeric(df["town_code"], errors="coerce").astype(int)
df["agas_code"] = pd.to_numeric(df["agas_code"], errors="coerce").fillna(0).astype(int)
df["date"] = pd.to_datetime(df["date"])

print(f"Raw rows loaded: {len(df):,}")

# ============================================================
# Build City_agas_code (same convention as preprocessing_corona_porat.ipynb)
# ============================================================
towns_with_real_agas = df.loc[df["agas_code"] != 0, "town_code"].unique()
df = df[~((df["agas_code"] == 0) & (df["town_code"].isin(towns_with_real_agas)))].copy()
df["City_agas_code"] = df["town_code"].astype(str) + "_" + df["agas_code"].astype(str)
df["is_city_aggregate"] = 0

# Synthesize missing city-level "_0" rows as the sum of a locality's sub-areas
existing_city_codes = set(df["City_agas_code"])
towns_missing_city_row = sorted(
    t for t in df["town_code"].unique() if f"{t}_0" not in existing_city_codes
)
sub_rows = df[df["town_code"].isin(towns_missing_city_row)]
city_agg = (
    sub_rows.groupby(["town_code", "date"], as_index=False)
    .agg({
        "accumulated_cases": "sum",
        "accumulated_diagnostic_tests": "sum",
        "accumulated_hospitalized": "sum",
        "town": "first",
    })
)
city_agg["City_agas_code"] = city_agg["town_code"].astype(str) + "_0"
city_agg["is_city_aggregate"] = 1

df = pd.concat([df, city_agg], ignore_index=True)
df = df.sort_values(["City_agas_code", "date"]).reset_index(drop=True)

print(f"Synthesized city rows added: {len(city_agg):,}")
print(f"Unique areas (City_agas_code) after key build: {df['City_agas_code'].nunique():,}")

# ============================================================
# Daily diffs over the FULL series (before filtering to the window,
# so the window's first day still gets a valid diff from the day before it)
# ============================================================
g = df.groupby("City_agas_code")
df["cases_daily"] = g["accumulated_cases"].diff()
df["tests_daily"] = g["accumulated_diagnostic_tests"].diff()
df["hosp_daily"] = g["accumulated_hospitalized"].diff()

for c in ["cases_daily", "tests_daily", "hosp_daily"]:
    df[c] = df[c].clip(lower=0)  # negative diffs = data corrections, not real counts

# ============================================================
# Adaptive infection rate (same method as preprocessing_corona_porat.ipynb)
# ============================================================
df["infection_rate"] = np.nan
g = df.groupby("City_agas_code", sort=False)

for lag in range(1, MAX_ADAPTIVE_WINDOW_DAYS + 1):
    prev_cases = g["accumulated_cases"].shift(lag)
    prev_tests = g["accumulated_diagnostic_tests"].shift(lag)
    prev_date = g["date"].shift(lag)

    cases_in_window = df["accumulated_cases"] - prev_cases
    tests_in_window = df["accumulated_diagnostic_tests"] - prev_tests
    actual_window_days = (df["date"] - prev_date).dt.days

    not_assigned = df["infection_rate"].isna()
    valid = (
        not_assigned
        & (tests_in_window >= MIN_TESTS_ADAPTIVE)
        & (cases_in_window >= 0)
        & (cases_in_window <= tests_in_window)
        & (actual_window_days >= 1)
        & (actual_window_days <= MAX_ADAPTIVE_WINDOW_DAYS)
    )
    df.loc[valid, "infection_rate"] = cases_in_window[valid] / tests_in_window[valid]

# ============================================================
# Restrict to the chosen window
# ============================================================
win = df[(df["date"] >= WINDOW_START) & (df["date"] <= WINDOW_END)].copy()
win = win.sort_values(["City_agas_code", "date"])
print(f"\nRows in window ({WINDOW_START.date()} - {WINDOW_END.date()}): {len(win):,}")
print(f"Areas present in window: {win['City_agas_code'].nunique():,}")


# ============================================================
# Per-area targets
# ============================================================
def area_row(area_df):
    area_df = area_df.sort_values("date")
    total_tests = area_df["tests_daily"].sum()
    total_cases = area_df["cases_daily"].sum()
    total_hosp = area_df["hosp_daily"].sum()

    ir = area_df["infection_rate"]
    mean_ir = ir.mean()

    peak_ir = np.nan
    slope_to_peak = np.nan
    if ir.notna().any():
        peak_idx = ir.idxmax()
        peak_ir = ir.loc[peak_idx]
        peak_date = area_df.loc[peak_idx, "date"]

        window_start_date = area_df["date"].iloc[0]
        days_from_start = (area_df["date"] - window_start_date).dt.days

        pre_peak_mask = area_df["date"] <= peak_date
        pre_peak_ir = ir.loc[pre_peak_mask]
        pre_peak_days = days_from_start.loc[pre_peak_mask]

        valid = pre_peak_ir.notna()
        if valid.sum() >= 2:
            slope_to_peak = np.polyfit(pre_peak_days[valid], pre_peak_ir[valid], 1)[0]

    severity = total_hosp / total_cases if total_cases > 0 else np.nan

    return pd.Series({
        "town": area_df["town"].iloc[0],
        "is_city_aggregate": area_df["is_city_aggregate"].iloc[0],
        "total_tests_in_window": total_tests,
        "total_cases_in_window": total_cases,
        "total_hospitalized_in_window": total_hosp,
        "mean_IR": mean_ir,
        "peak_IR": peak_ir,
        "slope_to_peak_IR": slope_to_peak,
        "severity_hosp_per_case": severity,
    })


table = win.groupby("City_agas_code").apply(area_row).reset_index()

# ============================================================
# Inclusion filter: keep an area only if all 4 target features are
# computable AND backed by a minimum amount of underlying data
# ============================================================
enough_tests = table["total_tests_in_window"] >= MIN_TESTS_IN_WINDOW
enough_cases = table["total_cases_in_window"] >= MIN_CASES_IN_WINDOW
has_all_4 = table[["mean_IR", "peak_IR", "slope_to_peak_IR", "severity_hosp_per_case"]].notna().all(axis=1)

print(f"\nAreas with any data in window: {len(table):,}")
print(f"  meet minimum tests (>= {MIN_TESTS_IN_WINDOW}): {enough_tests.sum():,}")
print(f"  meet minimum cases (>= {MIN_CASES_IN_WINDOW}): {enough_cases.sum():,}")
print(f"  all 4 features computable (no NaN): {has_all_4.sum():,}")

keep = enough_tests & enough_cases & has_all_4
final_table = table[keep].copy()
print(f"  FINAL kept (all conditions): {len(final_table):,}")
print(f"  dropped: {len(table) - len(final_table):,}")

out_cols = [
    "City_agas_code", "town", "is_city_aggregate",
    "mean_IR", "peak_IR", "slope_to_peak_IR", "severity_hosp_per_case",
    "total_tests_in_window", "total_cases_in_window", "total_hospitalized_in_window",
]
final_table = final_table[out_cols].sort_values("City_agas_code").reset_index(drop=True)

# ============================================================
# Save
# ============================================================
out_path = OUT_DIR / "wave2_ir_severity_table.csv"
final_table.to_csv(out_path, index=False, encoding="utf-8-sig")
print("\nSaved:", out_path)
print(final_table.head())

manifest = pd.DataFrame([{
    "file_path": str(out_path),
    "contents": (
        f"One row per City_agas_code (both is_city_aggregate=1 city rows and "
        f"sub-area rows, flagged separately). Window: {WINDOW_START.date()} - "
        f"{WINDOW_END.date()} (decline of wave 1 to end of the second lockdown). "
        f"4 model features: mean_IR, peak_IR, slope_to_peak_IR (speed to peak), "
        f"severity_hosp_per_case (cumulative hospitalized/cases over the whole "
        f"window). Only areas with >= {MIN_TESTS_IN_WINDOW} cumulative tests and "
        f">= {MIN_CASES_IN_WINDOW} cumulative cases in the window, and all 4 "
        f"features non-null, are included. Built directly from the raw file "
        f"(geographic-sum-per-day-ver_00859.csv) via scripts/build_wave2_ir_severity_table.py; "
        f"'<15' replaced with 0, infection_rate uses the adaptive-window method "
        f"(MIN_TESTS=30, up to 14 days)."
    ),
    "contents_he": (
        f"שורה אחת לכל City_agas_code (גם שורות עיר-שלמה is_city_aggregate=1 וגם "
        f"תתי-אזורים, מסומן בעמודה נפרדת). חלון: {WINDOW_START.date()} עד "
        f"{WINDOW_END.date()} (מדעיכת הגל הראשון עד סוף הסגר השני). 4 פיצ'רי מודל: "
        f"mean_IR, peak_IR, slope_to_peak_IR (קצב ההגעה לשיא), severity_hosp_per_case "
        f"(מאושפזים מצטבר חלקי מקרים מצטבר על פני כל החלון). רק אזורים עם "
        f">= {MIN_TESTS_IN_WINDOW} בדיקות מצטברות ו->= {MIN_CASES_IN_WINDOW} מקרים "
        f"מצטברים בחלון, וכל 4 הפיצ'רים לא-ריקים, נכללים. נבנה ישירות מהקובץ "
        f"הגולמי (geographic-sum-per-day-ver_00859.csv) דרך "
        f"scripts/build_wave2_ir_severity_table.py; '<15' הוחלף ב-0, infection_rate "
        f"בשיטה האדפטיבית (MIN_TESTS=30, עד 14 יום)."
    ),
}])
manifest_path = OUT_DIR / "wave2_ir_severity_manifest.csv"
manifest.to_csv(manifest_path, index=False, encoding="utf-8-sig")
print("Saved:", manifest_path)
