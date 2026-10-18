import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, precision_recall_curve, auc, classification_report

def add_forecast_features(df):
    """
    Builds lag features, rolling anomalies, and cyclical day-of-year features
    using only past/current information to prevent data leakage[span_1](start_span)[span_1](end_span).
    """
    df = df.sort_values(['city', 'date']).reset_index(drop=True)
    
    # 1. Cyclical day of year
    df['doy_sin'] = np.sin(2 * np.pi * df['dayofyear'] / 365.25)
    df['doy_cos'] = np.cos(2 * np.pi * df['dayofyear'] / 365.25)
    
    # 2. Lag features (previous day values per city)
    df['tmax_lag1'] = df.groupby('city')['tmax'].shift(1)
    df['tmax_lag2'] = df.groupby('city')['tmax'].shift(2)
    df['anomaly_lag1'] = df.groupby('city')['anomaly'].shift(1)
    
    # 3. Rolling means of anomaly (recent heat persistence)
    df['anomaly_roll3'] = df.groupby('city')['anomaly'].transform(
        lambda x: x.shift(1).rolling(window=3).mean()
    )
    df['anomaly_roll7'] = df.groupby('city')['anomaly'].transform(
        lambda x: x.shift(1).rolling(window=7).mean()
    )
    
    return df.dropna().reset_index(drop=True)

def chronological_split(df):
    """
    Splits data chronologically to prevent leakage:
    Train: 2000-2017, Validation: 2018-2020, Test: 2021-2024[span_2](start_span)[span_2](end_span)
    """
    years = df['date'].dt.year
    train_df = df[years.between(2000, 2017)]
    val_df = df[years.between(2018, 2020)]
    test_df = df[years.between(2021, 2024)]
    
    return train_df, val_df, test_df

def evaluate_temperature_models(train_df, val_df, test_df):
    """
    Trains temperature forecasting baselines and ML models (Ridge, Gradient Boosting).
    """
    features = ['tnormal', 'doy_sin', 'doy_cos', 'tmax_lag1', 'tmax_lag2', 'anomaly_lag1', 'anomaly_roll3', 'anomaly_roll7']
    target = 'tmax'
    
    X_train, y_train = train_df[features], train_df[target]
    X_val, y_val = val_df[features], val_df[target]
    X_test, y_test = test_df[features], test_df[target]
    
    # 1. Baseline: Persistence (tomorrow = today)
    pers_preds = test_df['tmax_lag1']
    print(f"Persistence Baseline RMSE: {np.sqrt(mean_squared_error(y_test, pers_preds)):.3f}")
    
    # 2. Ridge Regression Model
    ridge = Ridge(alpha=1.0)
    ridge.fit(X_train, y_train)
    ridge_preds = ridge.predict(X_test)
    print(f"Ridge Regression MAE: {mean_absolute_error(y_test, ridge_preds):.3f}, RMSE: {np.sqrt(mean_squared_error(y_test, ridge_preds)):.3f}")
    
    # 3. Gradient Boosting Regressor
    gbr = GradientBoostingRegressor(n_estimators=100, random_state=42)
    gbr.fit(X_train, y_train)
    gbr_preds = gbr.predict(X_test)
    print(f"Gradient Boosting MAE: {mean_absolute_error(y_test, gbr_preds):.3f}, RMSE: {np.sqrt(mean_squared_error(y_test, gbr_preds)):.3f}")
    
    return gbr_preds

def evaluate_heatwave_classifier(train_df, val_df, test_df):
    """
    Trains a binary classifier (Random Forest) for heatwave prediction,
    judged using Precision, Recall, F1, and PR-AUC[span_3](start_span)[span_3](end_span).
    """
    features = ['tnormal', 'anomaly_lag1', 'anomaly_roll3', 'anomaly_roll7', 'doy_sin', 'doy_cos']
    target = 'heatwave'
    
    X_train, y_train = train_df[features], train_df[target]
    X_test, y_test = test_df[features], test_df[target]
    
    # Random Forest with class weighting to handle severe class imbalance
    rf = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
    rf.fit(X_train, y_train)
    
    probs = rf.predict_proba(X_test)[:, 1]
    preds = rf.predict(X_test)
    
    precision, recall, _ = precision_recall_curve(y_test, probs)
    pr_auc = auc(recall, precision)
    
    print("\n--- Heatwave Classification Report ---")
    print(classification_report(y_test, preds, zero_division=0))
    print(f"PR-AUC Score: {pr_auc:.3f}")

if __name__ == '__main__':
    # Example execution flow for the forecasting phase
    df = pd.read_csv('labelled_dataset.csv', parse_dates=['date'])
    df_features = add_forecast_features(df)
    train, val, test = chronological_split(df_features)
    
    print("Evaluating Temperature Forecasting Models...")
    evaluate_temperature_models(train, val, test)
    
    print("\nEvaluating Heatwave Binary Classifier...")
    evaluate_heatwave_classifier(train, val, test)
