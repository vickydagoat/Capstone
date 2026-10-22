import pandas as pd
from scipy import stats
from pathlib import Path
from features import build_dataset

BASE = Path(__file__).parent
df = build_dataset(pd.read_csv(BASE / "imd_city_daily.csv", parse_dates=["date"]))

# 1. Is pre-monsoon Tmax (Mar-Jun) trending upward? (linear regression per city)
annual = df[df["month"].between(3, 6)].groupby(["year", "city"])["tmax"].mean().unstack()
for city in annual:
    r = stats.linregress(annual.index, annual[city])
    print(f"{city}: {r.slope * 10:+.2f} °C/decade, p = {r.pvalue:.3f}")

# 2. How well does the Kaggle dataset agree with IMD? (correlation, bias)
from data_loader import load_weather
k = load_weather("india_2000_2024_daily_weather.csv")
m = df.merge(k[["city", "date", "temperature_2m_max"]], on=["city", "date"])
for city, g in m.groupby("city"):
    r = stats.pearsonr(g["tmax"], g["temperature_2m_max"])[0]
    bias = (g["temperature_2m_max"] - g["tmax"]).mean()
    print(f"{city}: r = {r:.3f}, Kaggle - IMD bias = {bias:+.2f} °C")
