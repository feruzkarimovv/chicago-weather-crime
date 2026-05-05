# helpers for grabbing the raw data from the chicago portal and noaa

import os
import time
import requests

CRIME_URL = "https://data.cityofchicago.org/api/views/ijzp-q8t2/rows.csv?accessType=DOWNLOAD"
WEATHER_URL = "https://www.ncei.noaa.gov/data/global-historical-climatology-network-daily/access/USW00094846.csv"


def fetch_crime_data():
    out = "../data/raw/chicago_crime.csv"
    os.makedirs("../data/raw", exist_ok=True)
    print("downloading crime data")
    try:
        download(CRIME_URL, out)
    except Exception as e:
        print("first try failed:", e)
        print("waiting 5s and retrying...")
        time.sleep(5)
        download(CRIME_URL, out)
    return out


def fetch_weather_data():
    out = "../data/raw/chicago_weather.csv"
    os.makedirs("../data/raw", exist_ok=True)
    print("downloading weather data...")
    try:
        download(WEATHER_URL, out)
    except Exception as e:
        print("first try failed:", e)
        print("waiting 5s and retrying...")
        time.sleep(5)
        download(WEATHER_URL, out)
    return out


def download(url, out):
    r = requests.get(url, stream=True, timeout=120)
    r.raise_for_status()
    total = 0
    last = 0
    with open(out, "wb") as f:
        for chunk in r.iter_content(chunk_size=1024 * 1024):
            f.write(chunk)
            total += len(chunk)
            mb = total // (1024 * 1024)
            # crude progress every 200MB so the cell doesn't spam
            if mb - last >= 200:
                print(f"  {mb} MB")
                last = mb
    print(f"  done, {total // (1024*1024)} MB total")
