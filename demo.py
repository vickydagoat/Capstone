import pandas as pd
from cleaning import CITY_COORDS, clean_imd
from imd_loader import load_city_series
from features import add_tnormal, add_labels

def main():
    print("Starting Climate Intelligence Pipeline Demo...")
    
    # 1. Load city series data
    dfs = []
    for city, (lat, lon) in CITY_COORDS.items():
        try:
            df_city = load_city_series(city, lat, lon)
            dfs.append(df_city)
        except Exception as e:
            print(f"Could not load IMD data for {city}: {e}")
            
    if dfs:
        df_all = pd.concat(dfs, ignore_index=True)
        df_all = clean_imd(df_all)
        
        # 2. Engineer features and labels
        df_all = add_tnormal(df_all)
        df_all = add_labels(df_all)
        
        df_all.to_csv('labelled_dataset.csv', index=False)
        print("Pipeline complete. Saved 'labelled_dataset.csv'.")
        
        # Print label summary stats
        total_days = len(df_all)
        hw_days = (df_all['heatwave'] == 1).sum()
        print(f"Total city-days: {total_days}")
        print(f"Heatwave days: {hw_days} ({hw_days/total_days*100:.2f}%)")

if __name__ == '__main__':
    main()
