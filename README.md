# Chicago Weather and Crime

CS 210 final project. Looking at whether daily weather predicts daily crime in Chicago, 2018–2024.

**Final report:** `report/final_report.pdf`
**Demo video:** https://share.synthesia.io/a086ba38-2b84-4616-9538-0c814d2f0779
---

## What's in here

- Two real public datasets (Chicago crime + NOAA O'Hare weather), cleaned and merged.
- A SQLite database with three tables and six demo SQL queries (JOIN, CASE WHEN, window function, CTE).
- EDA with three statistical tests and 9 figures.
- Four predictors compared: mean baseline, day-of-week mean baseline, Ridge regression, Random Forest.
- Per-category Random Forest models across all six crime types.
- TimeSeriesSplit cross-validation and a paired t-test on per-fold MAE.

---

## Setup

Python 3 (3.10+ tested; built on 3.13).

```bash
git clone https://github.com/feruzkarimovv/chicago-weather-crime
cd chicago-weather-crime
pip install -r requirements.txt
```

---

## How to reproduce

The first notebook downloads the raw data (the crime CSV is ~1.9 GB and takes a few minutes). After that, run the notebooks in order:

1. `notebooks/01_data_collection.ipynb` — download Chicago crime + NOAA weather
2. `notebooks/02_data_cleaning.ipynb` — clean both datasets, aggregate daily, merge
3. `notebooks/03_database_setup.ipynb` — build SQLite + run 6 SQL queries
4. `notebooks/04_eda.ipynb` — 9 figures + Welch t-test, ANOVA, Pearson
5. `notebooks/05_modeling.ipynb` — 4 models, CV, diagnostic figures, per-category
6. `notebooks/06_results.ipynb` — pulls everything together for the report

Or open them all at once:

```bash
jupyter lab notebooks/
```

---

## Project structure

```
chicago-weather-crime/
├── README.md
├── requirements.txt
├── data/                         # raw + processed (raw not committed)
│   ├── raw/                      # CSVs from city + NOAA
│   ├── processed/merged.csv
│   └── chicago.db                # SQLite, three tables
├── notebooks/                    # 6 notebooks, run in order
├── src/                          # shared functions
│   ├── data_collection.py
│   ├── cleaning.py
│   ├── features.py
│   └── viz.py
├── outputs/
│   ├── figures/                  # 15 PNGs used in the report
│   ├── comparison.csv            # 4-model comparison
│   ├── per_category.csv          # per-category R²/MAE/RMSE
│   └── rf_model.pkl              # pickled main RF
└── report/
    ├── final_report.md           # markdown source
    └── final_report.pdf          # submitted deliverable
```

---

## Data

- **Chicago crime** — City of Chicago Open Data Portal, *Crimes – 2001 to Present.*
- **Weather** — NOAA NCEI Global Historical Climatology Network – Daily, station USW00094846 (Chicago O'Hare).

Raw data is not committed (the crime CSV is ~1.9 GB). The first notebook downloads it.

---

## Results

| Model | Test MAE | Test R² |
|---|---|---|
| Mean baseline | 74.95 | -0.589 |
| Day-of-week mean | 75.19 | -0.571 |
| Ridge regression | 40.49 | 0.456 |
| Random Forest | **39.37** | **0.463** |

Random Forest beats both baselines by ~47% on test MAE. Weather features collectively contribute ~14% of permutation importance; the 7-day rolling crime average dominates at 62%. Per-category, battery (R² = 0.40) and theft (0.27) are weather-sensitive; burglary and narcotics largely aren't.

Full discussion in `report/final_report.md`.

---

## Contact

Feruz Karimov | feruz.mit@gmail.com
