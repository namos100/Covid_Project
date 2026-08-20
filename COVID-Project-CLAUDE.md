# CLAUDE.md — COVID-19 Local Infection Prediction (Israel)

## Purpose of this file

Instructions, memory, and context for this project — for Claude Code, for
this chat, and for anyone else picking the project up. Put this file at the
root of the COVID project's own folder; it's a separate codebase from the
microbiome/aging project and should not be merged with that project's
CLAUDE.md.

This file mixes three kinds of content, and it matters which is which:
**established facts** (data really looks like this, this bug really
happened), **standing instructions** (how to work on this project going
forward), and **open/recommended items** (not yet decided or not yet built).
Where something is still open or just recommended, this file says so
explicitly — treat those as flexible, not as settled.

## Project identity & current status

- **Owners:** Porat Hanani and Naama Amos.
- **Canonical title:** "Predicting Local COVID-19 Infection Dynamics Across
  Geographic Locations in Israel Using Epidemiological and Socio-Demographic
  Data." Use this title unless scope or prediction target changes
  substantially.
- **Type:** started as a data-analysis-course final project; being revived
  (as of January 2026) with the intent of rebuilding it more rigorously and
  potentially developing it into a paper.
- **Geographic focus:** Israel. Initial case studies: Jerusalem and Tel Aviv.
  Long-term intent: expand to a national framework.
- **Target history:** hospitalization and mortality prediction were
  considered early on and dropped — available data was judged insufficient
  for reliable prediction. Current direction is infection-rate / local
  infection dynamics.
- **Status of old results:** the historical XGBoost/LSTM numbers below come
  from the earlier pipeline. Do not treat them as validated — the January
  2026 restart exists specifically to reproduce them under a more careful,
  leakage-checked pipeline before trusting the old conclusions.

## How to work on this project

- Be scientifically honest. Don't invent data sources, variables, geographic
  identifiers, model results, or performance values.
- Keep these visibly distinct in anything you write: decisions already made,
  current working assumptions, your recommendations, historical results, and
  genuinely open questions. Don't describe an expected result as a completed
  finding.
- Don't assume an older analysis is still valid until it's been reproduced
  under the current pipeline.
- Point out methodological weaknesses directly rather than just agreeing
  with a proposed approach.
- Keep the proposal, code, results, presentation, and conclusions internally
  consistent with each other.
- Don't interpret prediction as causation, and don't treat feature
  importance as evidence that a variable causes infection. Frame the model
  as a decision-support tool, not a replacement for epidemiologists or
  public-health authorities.
- Same practical habits as the microbiome project: paste the actual
  traceback rather than a paraphrase, give complete runnable code over
  fragments, make the smallest change that satisfies the request, and never
  claim code ran or a result holds unless it was actually run and inspected.

## Data sources & schema

### 1. Daily geographic COVID data

Raw file: `geographic-sum-per-day-ver_00859.csv` (Israeli government open
data — daily case/test/vaccination counts per city/sub-area).

Key columns: `town_code`, `agas_code`, `town`, `date`, `accumulated_cases`,
`new_cases_on_date`, `accumulated_recoveries`, `new_recoveries_on_date`,
`accumulated_hospitalized`, `new_hospitalized_on_date`, `accumulated_deaths`,
`new_deaths_on_date`, `accumulated_diagnostic_tests`,
`new_diagnostic_tests_on_date`, `accumulated_vaccination_first_dose`,
`new_vacc_first_dose_on_date`, `accumulated_vaccination_second_dose`,
`accumulated_vaccination_third_dose` (and matching `new_*` boolean-flag
columns).

**Known data quirk:** small counts are censored as the literal string
`'<15'` (Israeli data-privacy convention). Convention has been to replace
`'<15'` with `7` before converting to numeric — flag this assumption if it
matters for a given analysis rather than silently reapplying it.

### 2. Socio-demographic / census data

Source: 2022 Israeli population census (מפקד אוכלוסין 2022).

**Updated census data file:** `data/raw/mifkad_2.csv` — this is the current/
updated version of the census data (supersedes any earlier `mifkad`
file for this purpose).

**Data dictionary:** `data/raw/mifkad_data_dictionary.csv` documents what
each column in the census data means (Hebrew label, English label, group,
measurement type, unit, and modelling caveats per column) — consult it
before adding/interpreting any census column rather than guessing from the
name alone.

Key columns include: `LocNameHeb`, `LocalityCode`, `StatArea`,
`StatAreaCmb`, `pop_approx`, `change_pcnt`, `pop_density`, `religion`,
`ReligionHeb`, `sexRatio`, `inst_pcnt`, `Foreign_pcnt`, `age0_19_pcnt`,
`age20_64_pcnt`, `age65_pcnt`, `DependencyRatio`, `age_median`,
`m_age_median`, `w_age_median`, `married18_34_pcnt`, `married45_54_pcnt`,
`j_isr_pcnt`, `j_abr_pcnt`, `aliya2002_pcnt`, `aliya2010_pcnt`,
`israel_pcnt`, `asia_pcnt`, `africa_pcnt`, `europe_pcnt`, `america_pcnt`,
`MarriageAge_mdn` (+ gender-split variants). **Static** per city/sub-area —
repeated across every date for that location in a merged table, and must be
interpreted as background characteristics, not something that tracks
pandemic-era change (migration, employment shifts, behavior change, etc. are
all invisible to it).

`StatAreaCmb` sometimes lists combined sub-areas as a string like
`(16+11+12+17)` — reuse the existing `parse_statarea_cmb()` helper rather
than re-parsing this column from scratch.

### 3. The shared key: `City_agas_code`

Built as `LocalityCode + "_" + StatArea` (e.g. `'5000_0'` = Tel Aviv,
`'3000_0'` = Jerusalem). **`StatArea == 0` means "whole city," not a real
sub-area** — don't filter it out as a null/placeholder. Sub-areas within a
city are selected via e.g. `index.str.startswith('3000')`.

### Static vs. dynamic variables, and geographic harmonization

- Dynamic (changes daily/weekly): cases, infection rate, tests, positivity,
  vaccination, lockdown status, rolling averages, growth rates, lagged
  outcomes.
- Static (fixed or slow-changing): income, education, socio-economic
  classification, population density, demographic composition, census
  characteristics.
- The COVID dataset and the census dataset don't necessarily use identical
  geographic definitions. Don't force an exact join between incompatible
  units — defensible options are aggregating both to city level, a
  documented crosswalk, restricting to areas with reliable matches, or
  explicitly reporting unmatched areas (see "Missing-city resolution"
  below for the method already in use). Whatever resolution is chosen must
  stay consistent across target construction, features, training,
  validation, and interpretation.

## Known pitfalls already discovered in this codebase

- **Socio-economic features are constant within a single city.** Modeling
  one city alone means those columns have zero within-city variance and
  show ~zero feature importance — a real, already-diagnosed limitation, not
  a bug to patch. It's why the project moved toward panel-style modeling
  (multiple cities/areas together, city encoded as a feature or as an LSTM
  sequence dimension with static features repeated as "side inputs").
- **Division-based rates can produce `inf`.** `infection_rate` and similar
  ratio columns need explicit `.replace([inf, -inf], np.nan)` — this has
  happened before (Jerusalem infection rate coming out `inf` on some days).
- **`rolling(1)` is a no-op disguised as a feature.** If a "rolling" feature
  isn't behaving as expected, check the window size first.
- **Vaccination percentage columns have produced >100% coverage** in past
  plots — a units/scaling bug, not a real result. Sanity-check any
  vaccination-percentage plot against 0–100% before trusting it.
- **Suspiciously low error can mean leakage, not a good model.** The
  recovery-rate LSTM once produced MAE 0.0063 — flagged at the time as "too
  easy to predict, might be wrong." Investigate an unusually strong result
  (e.g. a lag feature leaking the target) before reporting it.
- **Full-country / all-cities modeling has hit memory limits** before
  (array allocation errors around ~1.3M-row scale); city count was capped
  to make it run. Plan for this explicitly (chunking, more efficient
  dtypes, sampling) when extending nationally rather than rediscovering the
  same error.
- **The historical Jerusalem XGBoost MAE (0.0038) and RMSE (0.0965) are
  oddly mismatched** — that combination suggests either a handful of
  predictions with very large errors, or the two metrics being computed on
  different arrays/scales. Don't reuse this number without re-deriving it
  under the rebuilt pipeline; if it reproduces, it needs an explanation in
  the write-up, not just a report.
- **COVID sub-area codes and 2022 census `StatArea` codes are not the same
  numbering scheme for a meaningful chunk of localities** — this is
  separate from and worse than the `StatAreaCmb` pooling issue (see below).
  Verified 2026-08-05: of 81 localities with sub-area breakdowns on both
  sides, 50 have any code mismatch, 30 have >30% of their COVID sub-areas
  with no matching census code, and 15 have **zero** overlap (e.g. Ness
  Ziona: COVID uses `1–8, 11`, census uses `12–33` for the same city; Beit
  Shemesh: COVID uses 2-digit codes, census uses 3-digit hierarchical
  codes). Whole-city rows (`area == 0`) always match; the mismatch is
  entirely at sub-area resolution. This hits the two case-study cities
  directly: **Jerusalem 63/148 (43%) and Tel Aviv 54/112 (48%) of sub-areas
  currently get no census features at all** after merging (`how="left"`
  leaves them NaN rather than dropping them, so they're easy to miss unless
  you check `_merge`/NaN counts explicitly). Current decision (as of
  2026-08-05): leave the merge as `left` with NaN for unmatched sub-areas,
  document the limitation, defer resolution. See "Open decisions" below —
  the "geographic crosswalk" item there is this, now quantified rather than
  hypothetical. Do not attempt to further parse/pool your way out of this;
  it isn't a parsing bug, it's two different area-delineation schemes.
  Reproduce via: compare `set(covid agas_code per town_code)` against
  `set(census StatArea per LocalityCode)` directly, not through
  `City_agas_code` string matching.

## General leakage & methodology checklist (check on every new analysis, not just once)

Temporal data leakage · geographic leakage · random splitting of
longitudinal observations · features using future information · inconsistent
geographic definitions · normalization fit before splitting · target leakage
· duplicate geography-date rows · wrong population denominators · many
zero/near-zero target values · shifts in testing availability, reporting
policy, variants, or vaccination coverage over time · overfitting to just
Jerusalem/Tel Aviv · overinterpreting feature importance · unfair
model-vs-model comparison · evaluation without a real baseline.

If any of these apply to a given analysis, say so explicitly rather than
letting it pass silently.

## Feature engineering

### Established conventions

- `infection_rate` from positive tests / tested; similarly
  `hospitalization_rate_per_100k`, `death_rate_per_100k`, and a
  recovery-rate/index from accumulated positive, accumulated recovered, and
  tested counts.
- `is_lockdown` boolean from a fixed list of Israeli lockdown date ranges:
  `[('2020-03-19','2020-04-24'), ('2020-09-18','2020-10-18'), ('2020-12-27','2021-02-06'), ('2021-07-29','2021-11-27')]`
  (converted to `pd.to_datetime`). Reuse this list rather than re-deriving it.
- Lag/rolling features (`infection_rate_lag1`, `infection_rate_rolling3`,
  etc.) computed **per city/area group**
  (`.groupby('City_agas_code')[...]`) — never globally across the whole
  dataframe, since these are independent time series per location.
- A log-transformed infection-rate target and a binary outbreak-prediction
  framing have both been explored as alternatives to raw regression.
- **Missing-city resolution** when merging COVID data with census data uses
  two explicit, manual strategies, not automatic imputation: (1) borrow a
  named neighboring city's socioeconomic values via a hand-built
  `neighbor_map` dict; (2) an explicit manual list of cities that can't be
  reasonably matched, dropped deliberately. Keep this explicit/reviewable
  approach rather than switching to automatic imputation.
- Every feature must represent information that would actually have been
  available on the prediction date — this includes being careful with
  revised/corrected case counts that weren't available in real time, and
  lagging vaccination variables since vaccination isn't expected to affect
  risk immediately.

### Candidate features (not all implemented yet — proposals, not facts)

Epidemiological lags at t-1/t-3/t-7/t-14 for rate, cases, positivity, test
volume · 7- and 14-day rolling mean and std, week-over-week difference/ratio,
recent acceleration · calendar features (day of week, week of year, month,
holiday, wave period) · vaccination coverage + lag + recent change + time
since milestone · lockdown status, restriction stage, days since
change · geographic/demographic features already in the census data.
Interactions may be worth exploring but shouldn't be added automatically
without a scientific reason.

## Modeling

Two model families used throughout: **XGBoost** and **LSTM**, applied both
per-city (Tel Aviv `'5000_0'`, Jerusalem `'3000_0'`) and per-sub-area within
a city. Train/test splits are chronological (`shuffle=False`) — referred to
in presentations as "walk-forward validation to respect temporal order."
That phrase has needed re-explaining more than once — when it comes up,
explain plainly what it means (train strictly precedes test in time) rather
than just reusing the phrase.

Model justification workflow used before jumping to ML: check autocorrelation
of the target, compare against a naive baseline (`shift(1)` persistence)
using MAE, and check feature correlations — the point being to demonstrate,
including in the presentation, that ML is actually justified before running
one.

**Required baselines** before any model is called successful: last observed
value, previous day's rate, previous week's rate, 7-day rolling mean, 14-day
rolling mean, and ideally a (regularized) linear regression. A model is not
"good" just because its error looks small — it has to beat these on
held-out future data.

**Historical results on record** (pre-2026-rebuild, NOT yet reproduced or
leakage-checked — see the mismatch flagged above):

| Model | City | MAE | RMSE |
|---|---|---|---|
| XGBoost | Jerusalem | 0.0038 | 0.0965 |
| XGBoost | Tel Aviv | 0.0028 | 0.0096 |
| LSTM | Jerusalem | 0.0786 | 0.1987 |
| LSTM | Tel Aviv | 0.0769 | 0.1475 |
| Naive baseline | — | ~0.0620 | — |

XGBoost has historically outperformed LSTM by a wide margin. Treat LSTM as a
stretch goal, not a priority — don't invest in it ahead of a complete,
reproducible baseline + XGBoost pipeline. If revisited: sequences must
preserve time order, must not contain future observations, and must not
cross improperly between geographic locations.

Feature importance: top-15 XGBoost importance and permutation importance
have both been used; SHAP has been requested specifically for LSTM.

## Validation requirements

- **Primary split must be chronological** — earliest dates train, later
  dates validate, latest dates test. Never randomly split rows for the
  primary evaluation; later-date observations can leak into training.
- **Rolling-origin evaluation** (train on periods 1–3 → test on 4; train on
  1–4 → test on 5; etc.) gives a more stable forecasting estimate than one
  fixed split, and is worth using for the rebuilt pipeline.
- **Geographic holdout** (train on some locations, test on unseen ones)
  measures generalization separately from future-date prediction — report
  it separately if used, since it's answering a different question.
- **Stratify evaluation** by city, epidemic wave, low- vs high-incidence
  period, and forecast horizon — a good average can hide poor performance
  exactly when it matters (during outbreaks).
- Metrics beyond MAE/RMSE worth considering: MASE, median absolute error,
  Spearman correlation, calibration; if an outbreak-classification target is
  ever used, sensitivity/specificity, ROC-AUC, precision-recall.

## Writing & framing conventions for proposal / paper / presentation

- Use the phrase **local COVID-19 infection dynamics** where it fits, and
  emphasize that national averages can hide real geographic variation.
- Describe Jerusalem and Tel Aviv as the initial case studies, with the
  broader framework intended to cover more locations across Israel.
- Never claim socio-demographic characteristics directly *cause* infection.
  Prefer: "may contribute predictive information," "may be associated
  with," "will be evaluated as a potential predictor," "may help explain
  differences in model performance."
- Don't promise the model will definitively identify future outbreaks.
  Prefer terms like "risk estimation," "forecasting," "elevated outbreak
  risk," "decision support."
- Preferred framing: *"This project develops and evaluates machine-learning
  models designed to forecast local COVID-19 infection dynamics across
  geographic locations in Israel."* Avoid: *"This project will accurately
  predict future outbreaks."*
- A strong conclusion connects: the local-forecasting problem → the
  limitation of national averages → integration of multiple data types →
  the ML approach → chronological/geographic validation → the expected
  scientific contribution → the practical public-health value → the
  limitations of the data.
- Literature anchors already used in the slides: Lucas et al. (LSTM +
  mobility data, US county-level forecasting), Chandra et al. (LSTM across
  Indian states), Du et al. (2023) — check these before rebuilding a
  literature slide from scratch.
- Presentation outline used to date: title & team, motivation & relevance,
  research objectives, literature review, data description, methodology,
  results, limitations, conclusions.

## Interpretability plan

XGBoost built-in importance, permutation importance, SHAP (overall and
dependence plots), city-specific and wave-specific importance. Key questions
to keep asking of any importance result: are recent infection values just
dominating everything? does vaccination add information beyond that? do
lockdown indicators help? do socio-demographic variables help *after*
recent case history is already in the model? do important predictors differ
between Jerusalem and Tel Aviv, or across waves? None of this is a causal
claim, however it comes out.

## Limitations to keep citing

- **Testing behavior**: confirmed cases reflect who got tested, not true
  infection counts — affected by policy, availability, awareness, and
  differences between communities.
- **Static census data**: doesn't reflect pandemic-era migration,
  employment, or behavior change.
- **Sparse local data** at fine geographic resolution: zero-heavy counts,
  small denominators, unstable rates, privacy-based suppression.
- **Non-stationarity**: variants, vaccination, waning immunity, and policy
  all shifted over the pandemic — a model trained on one wave may not
  transfer to another.
- **Generalization**: a model built on Jerusalem/Tel Aviv may not transfer
  to small towns, rural areas, or areas with different testing behavior or
  data quality, without explicit geographic validation.
- **No causal claims**: this is a predictive framework, not a causal design
  — it can't establish that income, education, density, or lockdowns
  directly cause a change in infection.

## Recommended repository structure (for the 2026 rebuild — not yet implemented)

```text
covid_project/
├── data/
│   ├── raw/
│   ├── interim/
│   ├── processed/
│   └── external/
├── scripts/
│   ├── 01_preprocessing.py
│   ├── 02_eda.py
│   ├── 03_feature_engineering.py
│   ├── 04_train_baselines.py
│   ├── 05_train_xgboost.py
│   ├── 06_train_lstm.py
│   └── 07_evaluate_models.py
├── notebooks/
├── src/
├── config/
├── outputs/
│   ├── figures/ tables/ models/ predictions/ logs/
├── reports/
└── CLAUDE.md
```

The earlier (pre-rebuild) code used a simpler ad hoc layout under
`C:\porat\data course final project\data\{raw,processed,outputs}` in
Jupyter, later ported into VS Code as `.py` scripts. Don't assume the
structure above is live until it's actually been set up — check first.

Preprocessing script responsibilities once built: load raw files without
modifying them, standardize column names/dates/geographic identifiers,
detect duplicate location-date rows, check chronological order, resolve
missing/infinite values, validate population values, merge daily + static
data, add lockdown indicators, generate leakage-safe features, and save one
canonical processed file (Parquet preferred) plus a preprocessing summary.

## Open decisions — don't silently assume answers

Exact target variable (count / rate / positivity / growth / outbreak
category) · prediction horizon · final geographic resolution · exact
dataset + date range · which vaccination variables are actually
available · how lockdown periods should be encoded going forward · canonical
census dataset · geographic crosswalk · population denominator · handling of
missing dates and of corrected case counts · handling of zero-heavy
targets · which baselines are mandatory · city-specific vs. pooled national
models · whether geographic holdout testing is in scope · which variables
would truly have been available in real time · whether LSTM is still worth
the effort · whether uncertainty intervals will be estimated · exact
required outputs for the course vs. for a future publication.

**Geographic crosswalk is no longer purely hypothetical** — see the
COVID-vs-census `StatArea` code mismatch documented under "Known pitfalls"
above (up to 43-48% of Jerusalem/Tel Aviv sub-areas unmatched). Still open:
whether to aggregate to city level, build a real crosswalk, restrict to
matched sub-areas, or keep deferring as currently decided.

## Current priorities (roadmap for the rebuild)

1. Inventory all available files and document each dataset/variable.
2. Choose the geographic unit and validate geographic matching.
3. Choose the exact target and a public-health-relevant forecast horizon.
4. Build the preprocessing script; save the canonical processed dataset.
5. Structured EDA.
6. Build the required simple baselines.
7. Implement chronological (and ideally rolling-origin) validation.
8. Reproduce the historical XGBoost result and investigate the metric
   mismatch flagged above.
9. Add interpretability (importance + SHAP).
10. Decide whether LSTM is worth pursuing further.
11. Expand beyond Jerusalem and Tel Aviv.
12. Rewrite the proposal/paper based on verified (not historical) results.

## Working conventions

- Windows/local paths in early work were under
  `C:\porat\data course final project\data\...` — expect the recommended
  `data/raw|processed|outputs` layout once the rebuild is underway.
- When something errors, work from the actual pasted traceback, not a
  guessed cause.
- Same expectations as the microbiome project: complete runnable code over
  fragments, smallest correct change, flag before removing anything, never
  claim a result without having actually run and inspected it.
