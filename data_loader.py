from pathlib import Path
import pandas as pd


def load_weather(filename):
    """Read the CSV that sits next to this script, parse dates, sort by city and date."""
    path = Path(__file__).parent / filename
    df = pd.read_csv(path, parse_dates=["date"])
    return df.sort_values(["city", "date"]).reset_index(drop=True)
