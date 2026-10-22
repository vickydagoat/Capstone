from data_loader import load_weather
from cleaning import quality_report, check_physical_validity, add_features

print("Script started")                       # if you see this, the file is running

df = load_weather("india_2000_2024_daily_weather.csv")
quality_report(df)
check_physical_validity(df)
df = add_features(df)

print("\nFirst rows after feature engineering:")
print(df.head())
print("\nColumns:", list(df.columns))
