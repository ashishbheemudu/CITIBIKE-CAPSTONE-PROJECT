
import pandas as pd
import numpy as np
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor
import joblib
import os
import logging
import gc
from sklearn.preprocessing import StandardScaler

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("SimpleTrainerV4")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "v1_core", "final_station_demand_robust_features.parquet")
MODELS_DIR = os.path.join(BASE_DIR, "models")

def engineer_features(df):
    """Standard Feature Engineering (In-Memory)"""
    df = df.copy()
    # Temporal
    df['hour'] = df['time'].dt.hour
    df['day'] = df['time'].dt.day
    df['month'] = df['time'].dt.month
    df['year'] = df['time'].dt.year
    df['day_of_week'] = df['time'].dt.dayofweek
    df['quarter'] = df['time'].dt.quarter
    
    # Cyclical
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24).astype(np.float32)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24).astype(np.float32)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12).astype(np.float32)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12).astype(np.float32)
    df['day_sin'] = np.sin(2 * np.pi * df['day'] / 31).astype(np.float32)
    df['day_cos'] = np.cos(2 * np.pi * df['day'] / 31).astype(np.float32)

    # Logic
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    df['is_holiday'] = 0 
    df['days_to_holiday'] = 7
    df['days_since_holiday'] = 7
    df['is_morning_rush'] = ((df['hour'] >= 7) & (df['hour'] <= 9)).astype(int)
    df['is_evening_rush'] = ((df['hour'] >= 17) & (df['hour'] <= 19)).astype(int)
    df['is_rush_hour'] = (df['is_morning_rush'] | df['is_evening_rush']).astype(int)
    df['is_business_hours'] = ((df['hour'] >= 9) & (df['hour'] <= 17) & (df['day_of_week'] < 5)).astype(int)
    df['time_of_day'] = pd.cut(df['hour'], bins=[-1, 6, 12, 18, 24], labels=[0, 1, 2, 3]).astype(int)
    df['season'] = (df['month'] % 12 // 3).astype(int)
    df['week_of_year'] = df['time'].dt.isocalendar().week.astype(int)
    df['is_month_start'] = df['time'].dt.is_month_start.astype(int)
    df['is_month_end'] = df['time'].dt.is_month_end.astype(int)
    epoch_start = pd.Timestamp('2019-01-01')
    df['days_since_epoch'] = (df['time'] - epoch_start).dt.days

    # Lags & Rolling
    g = df.groupby('station_name')['pickups']
    
    for lag in [1, 2, 3, 6, 12, 24, 48, 168]:
        df[f'lag_{lag}h'] = g.shift(lag).astype(np.float32)
        
    for w in [4, 12, 24, 168]:
        shifted = g.shift(1)
        r = shifted.rolling(window=w, min_periods=1)
        df[f'roll_mean_{w}h'] = r.mean().astype(np.float32)
        df[f'roll_std_{w}h'] = r.std().fillna(0).astype(np.float32)
        df[f'roll_min_{w}h'] = r.min().astype(np.float32)
        df[f'roll_max_{w}h'] = r.max().astype(np.float32)
        
    df['ema_24h'] = g.transform(lambda x: x.ewm(span=24, adjust=False).mean().shift(1)).astype(np.float32)

    df.dropna(inplace=True)
    return df

def train_v4():
    logger.info("🚀 Starting V4 Simple Training (Top 15 Stations)...")
    
    # 1. Load & Filter
    df_raw = pd.read_parquet(DATA_PATH, columns=['time', 'station_name', 'pickups', 'temp', 'prcp', 'wspd'])
    top_stations = df_raw['station_name'].value_counts().head(15).index
    df_raw = df_raw[df_raw['station_name'].isin(top_stations)].copy()
    df_raw.sort_values(['station_name', 'time'], inplace=True)
    logger.info(f"✅ Loaded {len(df_raw)} rows (Top 15).")

    # 2. Engineer Features
    df = engineer_features(df_raw)
    del df_raw
    gc.collect()
    logger.info(f"✅ Features Engineered. Shape: {df.shape}")

    # 3. Prepare X, y
    feature_cols = [c for c in df.columns if c not in ['pickups', 'station_name', 'time']]
    X = df[feature_cols].values.astype(np.float32)
    y = df['pickups'].values.astype(np.float32).reshape(-1, 1)
    
    # Save Feature Names for prediction service alignment
    import json
    with open(os.path.join(MODELS_DIR, "feature_names.json"), 'w') as f:
        json.dump(feature_cols, f)
    logger.info(f"✅ Saved {len(feature_cols)} feature names to feature_names.json")
    
    del df
    gc.collect()

    # 4. Scale
    logger.info("🔄 Scaling Data...")
    scaler_tree = StandardScaler()
    X_scaled = scaler_tree.fit_transform(X)
    scaler_y = StandardScaler()
    y_scaled = scaler_y.fit_transform(y)
    
    joblib.dump(scaler_tree, os.path.join(MODELS_DIR, "scaler_tree_server.save"))
    joblib.dump(scaler_y, os.path.join(MODELS_DIR, "scaler_y_server.save"))
    
    # 5. Train XGBoost
    logger.info("🏋️ Training XGBoost...")
    xgb_model = xgb.XGBRegressor(
        objective='reg:squarederror', n_estimators=100, learning_rate=0.05, max_depth=7, n_jobs=-1
    )
    xgb_model.fit(X_scaled, y_scaled)
    xgb_model.get_booster().save_model(os.path.join(MODELS_DIR, "xgb_server.json"))
    logger.info("✅ XGBoost Saved.")

    # 6. Train LightGBM
    logger.info("🏋️ Training LightGBM...")
    lgb_model = lgb.LGBMRegressor(n_estimators=100, learning_rate=0.05, n_jobs=-1)
    lgb_model.fit(X_scaled, y_scaled.ravel())
    lgb_model.booster_.save_model(os.path.join(MODELS_DIR, "lgb_server.txt"))
    logger.info("✅ LightGBM Saved.")

    # 7. Train CatBoost
    logger.info("🏋️ Training CatBoost...")
    cb_model = CatBoostRegressor(iterations=100, learning_rate=0.05, depth=6, allow_writing_files=False, verbose=False)
    cb_model.fit(X_scaled, y_scaled.ravel())
    cb_model.save_model(os.path.join(MODELS_DIR, "cb_server.cbm"))
    logger.info("✅ CatBoost Saved.")
    
    logger.info("🎉 V4 Simple Training Complete.")

if __name__ == "__main__":
    train_v4()
