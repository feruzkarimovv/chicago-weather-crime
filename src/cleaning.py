# cleaning helpers for crime + weather

import pandas as pd


def clean_crime(df):
    df = df.copy()

    # raw "Date" col is like "07/29/2022 03:39:00 AM" - faster if we give it the format
    df["date"] = pd.to_datetime(df["Date"], format="%m/%d/%Y %I:%M:%S %p")
    df["date_only"] = df["date"].dt.date

    # drop nulls
    df = df.dropna(subset=["date", "Primary Type"])

    # dedupe on the city's unique id
    df = df.drop_duplicates(subset=["ID"])

    # filter to 2018-2024
    df = df[(df["date_only"] >= pd.to_datetime("2018-01-01").date()) &
            (df["date_only"] <= pd.to_datetime("2024-12-31").date())]

    # normalize categories
    df["Primary Type"] = df["Primary Type"].str.upper()

    # bool casts (pandas reads these as actual bools usually but be safe)
    df["Arrest"] = df["Arrest"].astype(bool)
    df["Domestic"] = df["Domestic"].astype(bool)

    df = df.rename(columns={
        "ID": "id",
        "Primary Type": "primary_type",
        "Description": "description",
        "Arrest": "arrest",
        "Domestic": "domestic",
        "Latitude": "latitude",
        "Longitude": "longitude",
    })
    df = df.drop(columns=["Date"])
    return df


def aggregate_crime_daily(df):
    cats = ["THEFT", "BATTERY", "BURGLARY", "ASSAULT", "NARCOTICS", "CRIMINAL DAMAGE"]

    daily = df.groupby("date_only").size().rename("total_crimes").to_frame()

    for cat in cats:
        col = cat.lower().replace(" ", "_")
        daily[col] = df[df["primary_type"] == cat].groupby("date_only").size()

    daily["arrest_count"] = df[df["arrest"]].groupby("date_only").size()
    daily["domestic_count"] = df[df["domestic"]].groupby("date_only").size()

    daily = daily.fillna(0).astype(int)
    daily = daily.reset_index().rename(columns={"date_only": "date"})
    daily["date"] = pd.to_datetime(daily["date"])
    return daily


def clean_weather(df):
    df = df.copy()
    df["DATE"] = pd.to_datetime(df["DATE"])

    # filter to our window first so we don't waste work on 70 years of data
    df = df[(df["DATE"] >= "2018-01-01") & (df["DATE"] <= "2024-12-31")]

    keep = ["DATE", "TMAX", "TMIN", "PRCP", "AWND", "SNOW", "SNWD"]
    df = df[keep].copy()

    # noaa: tmax/tmin in tenths of C -> divide by 10, then add F columns
    df["tmax_c"] = df["TMAX"] / 10
    df["tmin_c"] = df["TMIN"] / 10
    df["tmax_f"] = df["tmax_c"] * 9 / 5 + 32
    df["tmin_f"] = df["tmin_c"] * 9 / 5 + 32

    # prcp in tenths of mm
    df["prcp_mm"] = df["PRCP"] / 10
    # awnd in tenths of m/s
    df["awnd_ms"] = df["AWND"] / 10

    # snow / snwd already in mm
    df["snow_mm"] = df["SNOW"]
    df["snwd_mm"] = df["SNWD"]

    # chicago is mostly snow free, so NaN means 0 here
    df["snow_mm"] = df["snow_mm"].fillna(0)
    df["snwd_mm"] = df["snwd_mm"].fillna(0)

    # forward-fill tmax/tmin up to 2 days, drop the rest
    for c in ["tmax_c", "tmin_c", "tmax_f", "tmin_f"]:
        df[c] = df[c].ffill(limit=2)
    df = df.dropna(subset=["tmax_c", "tmin_c"])

    # awnd: median impute if missingness is small
    miss = df["awnd_ms"].isna().mean()
    if miss < 0.05:
        df["awnd_ms"] = df["awnd_ms"].fillna(df["awnd_ms"].median())

    df = df.rename(columns={"DATE": "date"})
    out_cols = ["date", "tmax_c", "tmin_c", "tmax_f", "tmin_f",
                "prcp_mm", "snow_mm", "snwd_mm", "awnd_ms"]
    return df[out_cols]
