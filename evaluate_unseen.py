
import pandas as pd
import numpy as np
import joblib
import json
import os
import glob
from datetime import datetime
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor
from meteostat import Point, Daily
import warnings
warnings.filterwarnings('ignore')

# ---------------------------------------------------------
# CONFIG
# ---------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "backend", "models")
NEW_DATA_DIR = os.path.join(BASE_DIR, "new months")

def load_models():
    print("🚀 Loading models...")
    models = {}
    
    # Feature names
    with open(os.path.join(MODELS_DIR, "feature_names.json"), 'r') as f:
        feature_names = json.load(f)
        
    # Scalers
    scaler_tree = joblib.load(os.path.join(MODELS_DIR, "scaler_tree.save"))
    scaler_y = joblib.load(os.path.join(MODELS_DIR, "scaler_y.save"))
    
    # XGBoost
    models['xgb'] = xgb.XGBRegressor()
    models['xgb'].load_model(os.path.join(MODELS_DIR, "xgb.json"))
    
    # LightGBM
    models['lgb'] = joblib.load(os.path.join(MODELS_DIR, "lgb.pkl"))
    
    # CatBoost
    models['cb'] = CatBoostRegressor()
    models['cb'].load_model(os.path.join(MODELS_DIR, "cb.cbm"))
    
    # Ensemble weights
    with open(os.path.join(MODELS_DIR, "ensemble_config.json"), 'r') as f:
        config = json.load(f)
        weights = config.get('weights', {'xgb': 0.33, 'lgb': 0.33, 'cb': 0.34})
        
    print("✅ Models loaded.")
    return models, weights, feature_names, scaler_tree, scaler_y

def load_process_data():
    print("📂 Finding new data files...")
    
    # Find all CSVs in subfolders
    files = glob.glob(os.path.join(NEW_DATA_DIR, "**", "*.csv"), recursive=True)
    # Filter for tripdata, exclude JC (Jersey City) if we want pure NYC, or include if model is robust
    # The user said "Sep, Oct, Nov". Let's try to include all found relevant files.
    trip_files = [f for f in files if "tripdata" in f and "JC-" not in os.path.basename(f)]
    
    print(f"found {len(trip_files)} NYC trip files")
    for f in trip_files:
        print(f" - {os.path.basename(f)}")
        
    if not trip_files:
        print("⚠️ No NYC files found! Checking for JC (Jersey City) files...")
        trip_files = [f for f in files if "tripdata" in f]
        print(f"Found {len(trip_files)} total files including JC.")

    dfs = []
    for f in trip_files:
        try:
            print(f"Reading {os.path.basename(f)}...")
            # Read only essential cols
            df = pd.read_csv(f, usecols=['started_at', 'start_station_name'])
            dfs.append(df)
        except Exception as e:
            print(f"Error reading {f}: {e}")
            
    if not dfs:
        raise ValueError("No data loaded")
        
    print("Combining data...")
    full_df = pd.concat(dfs, ignore_index=True)
    full_df['started_at'] = pd.to_datetime(full_df['started_at'])
    
    # Filter dates to new period (Sep 2025 onwards) just to be sure
    full_df = full_df[full_df['started_at'] >= '2025-09-01']
    
    return full_df

def aggregate_hourly(df):
    print("⌛ Aggregating to hourly demand...")
    # Floor to hour
    df['hour'] = df['started_at'].dt.floor('H')
    
    # Group by station and hour
    hourly = df.groupby(['start_station_name', 'hour']).size().reset_index(name='pickups')
    hourly.rename(columns={'start_station_name': 'station_name', 'hour': 'time'}, inplace=True)
    
    return hourly

def fetch_weather(dates):
    print("☁️ Fetching weather data...")
    # New York coordinates
    nyc = Point(40.7128, -74.0060)
    
    start = min(dates)
    end = max(dates)
    
    try:
        data = Daily(nyc, start, end).fetch()
        # Create dictionary
        weather_map = {}
        for index, row in data.iterrows():
            date_str = index.strftime('%Y-%m-%d')
            weather_map[date_str] = {
                'temp': row['tavg'],
                'prcp': row['prcp'],
                'wspd': row['wspd']
            }
        return weather_map
    except Exception as e:
        print(f"Weather error: {e}")
        return {}

def generate_features(df, feature_names):
    print("🛠️ Generating features (Vectorized)...")
    
    # Sort for lag calculation
    df = df.sort_values(['station_name', 'time'])
    
    # 1. Weather
    # We need to map weather. For now, use average/default to avoid API complexity if it fails
    # Ideally use the fetch_weather
    unique_dates = df['time'].dt.date.unique()
    weather_map = fetch_weather(unique_dates)
    
    def get_weather(row, metric, default):
        date_str = row['time'].strftime('%Y-%m-%d')
        if date_str in weather_map:
            val = weather_map[date_str].get(metric)
            return val if not pd.isna(val) else default
        return default
        
    # Vectorized weather mapping is faster
    # Create temp dataframe for weather
    weather_data = []
    for d in unique_dates:
        d_str = d.strftime('%Y-%m-%d')
        w = weather_map.get(d_str, {'temp': 15.0, 'prcp': 0.0, 'wspd': 5.0})
        weather_data.append({
            'date': d,
            'temp': w.get('temp', 15.0),
            'prcp': w.get('prcp', 0.0),
            'wspd': w.get('wspd', 5.0)
        })
    w_df = pd.DataFrame(weather_data)
    w_df['date'] = pd.to_datetime(w_df['date']).dt.date
    
    df['date_only'] = df['time'].dt.date
    df = df.merge(w_df, left_on='date_only', right_on='date', how='left')
    
    # Fill missing
    df['temp'] = df['temp'].fillna(15.0)
    df['prcp'] = df['prcp'].fillna(0.0)
    df['wspd'] = df['wspd'].fillna(5.0)
    
    # 2. Temporal
    df['hour'] = df['time'].dt.hour
    df['day'] = df['time'].dt.day
    df['month'] = df['time'].dt.month
    df['year'] = df['time'].dt.year
    df['day_of_week'] = df['time'].dt.dayofweek
    df['quarter'] = df['time'].dt.quarter
    
    # Cyclical
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    df['day_sin'] = np.sin(2 * np.pi * df['day'] / 31)
    df['day_cos'] = np.cos(2 * np.pi * df['day'] / 31)
    
    # Boolean
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    # Simple logic for holidays (simplified)
    import holidays
    us_holidays = holidays.US()
    df['is_holiday'] = df['time'].apply(lambda x: 1 if x in us_holidays else 0)
    
    df['is_morning_rush'] = df['hour'].apply(lambda x: 1 if 7 <= x <= 9 else 0)
    df['is_evening_rush'] = df['hour'].apply(lambda x: 1 if 17 <= x <= 19 else 0)
    df['is_rush_hour'] = df[['is_morning_rush', 'is_evening_rush']].max(axis=1)
    df['is_business_hours'] = ((df['hour'] >= 9) & (df['hour'] <= 17) & (df['day_of_week'] < 5)).astype(int)
    
    # Time of day
    def get_tod(h):
        if h < 6: return 0
        if h < 12: return 1
        if h < 18: return 2
        return 3
    df['time_of_day'] = df['hour'].apply(get_tod)
    
    # Season
    def get_season(m):
        if m in [12, 1, 2]: return 0
        if m in [3, 4, 5]: return 1
        if m in [6, 7, 8]: return 2
        return 3
    df['season'] = df['month'].apply(get_season)
    
    df['week_of_year'] = df['time'].dt.isocalendar().week.astype(int)
    df['is_month_start'] = (df['day'] <= 3).astype(int)
    df['is_month_end'] = (df['day'] >= 28).astype(int)
    
    epoch = pd.Timestamp('2020-01-01')
    df['days_since_epoch'] = (df['time'] - epoch).dt.total_seconds() / 86400
    
    # Placeholder for complicated holiday distances
    df['days_to_holiday'] = 7
    df['days_since_holiday'] = 7
    
    # 3. Lags & Rolling
    # We must ensure the index is continuous in time for each station to use shift()
    # But real data has gaps (no trips at 3am).
    # To do this correctly:
    # reindex each station to full hourly frequency
    print("  ... Reindexing for time continuity (this takes RAM)...")
    
    # Filter to ONLY stations that exist in the original training data
    import os
    parquet_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "backend", "data", "v1_core", "final_station_demand_robust_features.parquet")
    training_df = pd.read_parquet(parquet_path, columns=['station_name'])
    training_stations = set(training_df['station_name'].unique())
    
    # Filter new data to only include training stations
    df = df[df['station_name'].isin(training_stations)].copy()
    all_stations = df['station_name'].unique()
    print(f"  Filtered to {len(all_stations)} stations (same as historical training data)")
    
    # Reindex
    min_time = df['time'].min()
    max_time = df['time'].max()
    full_range = pd.date_range(min_time, max_time, freq='H')
    
    combos = pd.MultiIndex.from_product([all_stations, full_range], names=['station_name', 'time'])
    df_full = df.set_index(['station_name', 'time']).reindex(combos, fill_value=0).reset_index()
    
    # Must restore features lost by fillna(0) or fill with correct values
    # Actually, simpler to regenerate calculated features after reindex
    # But 'pickups' is 0 for gaps.
    # We need to re-map raw features (temp, hour, etc) for the new rows.
    
    # Recalculate basic time features for all rows
    df_full['hour'] = df_full['time'].dt.hour
    df_full['day'] = df_full['time'].dt.day
    df_full['month'] = df_full['time'].dt.month
    df_full['year'] = df_full['time'].dt.year
    df_full['day_of_week'] = df_full['time'].dt.dayofweek
    df_full['date_only'] = df_full['time'].dt.date
    
    # Remerge weather
    df_full = df_full.drop(columns=['temp', 'prcp', 'wspd'], errors='ignore')
    df_full = df_full.merge(w_df, left_on='date_only', right_on='date', how='left')
    df_full['temp'] = df_full['temp'].fillna(15.0)
    df_full['prcp'] = df_full['prcp'].fillna(0.0)
    df_full['wspd'] = df_full['wspd'].fillna(5.0)
    
    # Recalculate derived
    # (Copy-paste logic for brevity or wrap in func - redoing essential ones)
    df_full['hour_sin'] = np.sin(2 * np.pi * df_full['hour'] / 24)
    df_full['hour_cos'] = np.cos(2 * np.pi * df_full['hour'] / 24)
    df_full['month_sin'] = np.sin(2 * np.pi * df_full['month'] / 12)
    df_full['month_cos'] = np.cos(2 * np.pi * df_full['month'] / 12)
    df_full['day_sin'] = np.sin(2 * np.pi * df_full['day'] / 31)
    df_full['day_cos'] = np.cos(2 * np.pi * df_full['day'] / 31)
    
    df_full['is_weekend'] = (df_full['day_of_week'] >= 5).astype(int)
    # ... assuming others re-calculated or preserved if needed ...
    # For brevity, let's fill key ones
    df_full['is_holiday'] = df_full['time'].apply(lambda x: 1 if x in us_holidays else 0)
    df_full['days_to_holiday'] = 7
    df_full['days_since_holiday'] = 7
    df_full['quarter'] = (df_full['month'] - 1) // 3 + 1
    df_full['is_morning_rush'] = df_full['hour'].apply(lambda x: 1 if 7 <= x <= 9 else 0)
    df_full['is_evening_rush'] = df_full['hour'].apply(lambda x: 1 if 17 <= x <= 19 else 0)
    df_full['is_rush_hour'] = df_full[['is_morning_rush', 'is_evening_rush']].max(axis=1)
    df_full['is_business_hours'] = ((df_full['hour'] >= 9) & (df_full['hour'] <= 17) & (df_full['day_of_week'] < 5)).astype(int)
    df_full['time_of_day'] = df_full['hour'].apply(get_tod)
    df_full['season'] = df_full['month'].apply(get_season)
    df_full['week_of_year'] = df_full['time'].dt.isocalendar().week.astype(int)
    df_full['is_month_start'] = (df_full['day'] <= 3).astype(int)
    df_full['is_month_end'] = (df_full['day'] >= 28).astype(int)
    df_full['days_since_epoch'] = (df_full['time'] - epoch).dt.total_seconds() / 86400
    df_full['ema_24h'] = 0.0 # Placeholder
    
    # Pickups is correct (0 for missing hours)
    # Since we re-indexed, 'pickups' column might be NaN for new rows if we didn't fill_value=0 in reindex?
    # Actually reindex(combos, fill_value=0) fills ALL cols with 0. That's bad for 'station_name' but station_name is in index.
    # The 'pickups' should be 0.
    if 'pickups' not in df_full.columns:
        # If pickups was lost in merge/reindex, we need to recover it
        # Actually our aggregate_hourly result had 'pickups'.
        # Merging back to df_full
        # Let's simple way:
        # Use shift() on the sorted grouped df
        pass
    
    # Calculate LAGS
    print("  ... Calculating lags...")
    df_full = df_full.sort_values(['station_name', 'time'])
    g = df_full.groupby('station_name')['pickups']
    
    df_full['lag_1h'] = g.shift(1).fillna(0)
    df_full['lag_2h'] = g.shift(2).fillna(0)
    df_full['lag_3h'] = g.shift(3).fillna(0)
    df_full['lag_6h'] = g.shift(6).fillna(0)
    df_full['lag_12h'] = g.shift(12).fillna(0)
    df_full['lag_24h'] = g.shift(24).fillna(0)
    df_full['lag_48h'] = g.shift(48).fillna(0)
    df_full['lag_168h'] = g.shift(168).fillna(0)
    
    # Rolling
    df_full['roll_mean_4h'] = g.rolling(4).mean().reset_index(0, drop=True).fillna(0)
    df_full['roll_std_4h'] = g.rolling(4).std().reset_index(0, drop=True).fillna(0)
    df_full['roll_min_4h'] = g.rolling(4).min().reset_index(0, drop=True).fillna(0)
    df_full['roll_max_4h'] = g.rolling(4).max().reset_index(0, drop=True).fillna(0)
    
    df_full['roll_mean_12h'] = g.rolling(12).mean().reset_index(0, drop=True).fillna(0)
    df_full['roll_std_12h'] = g.rolling(12).std().reset_index(0, drop=True).fillna(0)
    df_full['roll_min_12h'] = g.rolling(12).min().reset_index(0, drop=True).fillna(0)
    df_full['roll_max_12h'] = g.rolling(12).max().reset_index(0, drop=True).fillna(0)
    
    df_full['roll_mean_24h'] = g.rolling(24).mean().reset_index(0, drop=True).fillna(0)
    df_full['roll_std_24h'] = g.rolling(24).std().reset_index(0, drop=True).fillna(0)
    df_full['roll_min_24h'] = g.rolling(24).min().reset_index(0, drop=True).fillna(0)
    df_full['roll_max_24h'] = g.rolling(24).max().reset_index(0, drop=True).fillna(0)
    
    df_full['roll_mean_168h'] = g.rolling(168).mean().reset_index(0, drop=True).fillna(0)
    df_full['roll_std_168h'] = g.rolling(168).std().reset_index(0, drop=True).fillna(0)
    df_full['roll_min_168h'] = g.rolling(168).min().reset_index(0, drop=True).fillna(0)
    df_full['roll_max_168h'] = g.rolling(168).max().reset_index(0, drop=True).fillna(0)
    
    df_full['ema_24h'] = g.ewm(span=24, adjust=False).mean().reset_index(0, drop=True).fillna(0)
    
    # Drop first 168 hours (7 days) because lags are invalid
    print("  ... Dropping first 7 days (warmup)...")
    df_final = df_full[df_full['time'] >= df_full['time'].min() + pd.Timedelta(days=7)]
    
    return df_final

def evaluate():
    # 1. Load Data
    raw_df = load_process_data()
    hourly_df = aggregate_hourly(raw_df)
    
    print(f"Stats: {len(hourly_df)} hourly records from {hourly_df['time'].min()} to {hourly_df['time'].max()}")
    
    # 2. Load Models
    models, weights, feature_names, scaler_tree, scaler_y = load_models()
    
    # 3. Features
    test_df = generate_features(hourly_df, feature_names)
    
    # Select feature columns in order
    X = test_df[feature_names]
    y_true = test_df['pickups'].values
    
    print(f"Testing on {len(X)} samples...")
    
    # 4. Scale
    X_scaled = scaler_tree.transform(X)
    
    # 5. Predict
    final_pred_scaled = np.zeros(len(X))
    total_weight = 0
    
    print("🤖 Running predictions...")
    for name, model in models.items():
        w = weights.get(name, 0.33)
        print(f"  Running {name} (w={w})...")
        pred = model.predict(X_scaled)
        final_pred_scaled += pred * w
        total_weight += w
        
    final_pred_scaled /= total_weight
    
    # Inverse transform
    final_pred = scaler_y.inverse_transform(final_pred_scaled.reshape(-1, 1)).flatten()
    final_pred = np.maximum(final_pred, 0) # Clip negative
    
    # 6. Metrics
    from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
    
    r2 = r2_score(y_true, final_pred)
    mae = mean_absolute_error(y_true, final_pred)
    rmse = np.sqrt(mean_squared_error(y_true, final_pred))
    
    print("\n" + "="*50)
    print("RESULTS ON UNSEEN DATA (SEP-NOV 2025)")
    print("="*50)
    print(f"Samples: {len(y_true)}")
    print(f"R² Score: {r2:.4f}")
    print(f"MAE:      {mae:.4f}")
    print(f"RMSE:     {rmse:.4f}")
    print("="*50)
    
    # Save results
    results_df = pd.DataFrame({
        'time': test_df['time'],
        'station': test_df['station_name'],
        'actual': y_true,
        'predicted': final_pred
    })
    results_df.to_csv("unseen_test_results.csv", index=False)
    print("Saved results to unseen_test_results.csv")

if __name__ == "__main__":
    evaluate()
