# build the engineered feature columns from the merged daily dataset

import pandas as pd
import holidays


def build_features(merged):
    df = merged.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    # temporal
    df["day_of_week"] = df["date"].dt.dayofweek  # 0=mon, 6=sun
    df["month"] = df["date"].dt.month
    df["year"] = df["date"].dt.year
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)

    # us federal holidays
    us_hols = holidays.US(years=range(2018, 2025))
    df["is_holiday"] = df["date"].dt.date.isin(us_hols).astype(int)

    # temp range
    df["temp_range_f"] = df["tmax_f"] - df["tmin_f"]

    # bins
    df["is_rainy"] = (df["prcp_mm"] > 1).astype(int)
    df["is_hot"] = (df["tmax_f"] > 85).astype(int)
    df["is_cold"] = (df["tmin_f"] < 32).astype(int)

    # lag features
    df["tmax_f_lag1"] = df["tmax_f"].shift(1)
    df["prcp_mm_lag1"] = df["prcp_mm"].shift(1)

    # 7-day rolling avg of crime, shifted by 1 so it doesn't leak the label
    df["crime_7day_avg"] = df["total_crimes"].rolling(7).mean().shift(1)

    # special flags
    df["is_covid_year"] = (df["year"] == 2020).astype(int)
    df["is_nye"] = ((df["month"] == 12) & (df["date"].dt.day == 31)).astype(int)
    df["is_july4"] = ((df["month"] == 7) & (df["date"].dt.day == 4)).astype(int)

    return df
