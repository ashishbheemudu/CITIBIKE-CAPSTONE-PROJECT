
import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
import os
import logging
import gc
from sklearn.preprocessing import StandardScaler

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("ServerTrainer")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "v1_core", "final_station_demand_robust_features.parquet")
MODELS_DIR = os.path.join(BASE_DIR, "models")

def train_server_model():
    logger.info("🚀 Starting Server-Side Training Protocol...")
    
    # 1. Load Data
    if not os.path.exists(DATA_PATH):
        logger.error(f"❌ Data not found: {DATA_PATH}")
        return

    logger.info("📊 Loading Data (Columns subset for memory safety)...")
    # Load minimal columns first
    df = pd.read_parquet(DATA_PATH, columns=['time', 'station_name', 'pickups', 'temp', 'prcp', 'wspd'])
    logger.info(f"✅ Loaded {len(df)} rows.")

    # 2. Filter Top 15 Stations (Aggressive Memory Optimization)
    logger.info("✂️ Filtering to Top 15 Stations to preserve RAM...")
    top_stations = df['station_name'].value_counts().head(15).index
    df = df[df['station_name'].isin(top_stations)].copy()
    
    # Trigger GC
    import gc
    gc.collect()
    
    logger.info(f"✅ Filtered to {len(df)} rows (Top 15 stations).")
    
    # Sort for time-based feature generation
    df.sort_values(['station_name', 'time'], inplace=True)
    df.reset_index(drop=True, inplace=True)

    # 3. Vectorized Feature Engineering
    logger.info("🛠️ Engineering Features (Vectorized)...")
    
    # Temporal
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

    # Binary/Logic
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    # Simple holiday proxy (Manual list or simplified)
    # For speed, we skip complex holiday lookup on server or use simpler logic.
    # Leaving as 0 for now (Model will learn it's less important or capture via date)
    df['is_holiday'] = 0 
    df['days_to_holiday'] = 7
    df['days_since_holiday'] = 7
    
    df['is_morning_rush'] = ((df['hour'] >= 7) & (df['hour'] <= 9)).astype(int)
    df['is_evening_rush'] = ((df['hour'] >= 17) & (df['hour'] <= 19)).astype(int)
    df['is_rush_hour'] = (df['is_morning_rush'] | df['is_evening_rush']).astype(int)
    df['is_business_hours'] = ((df['hour'] >= 9) & (df['hour'] <= 17) & (df['day_of_week'] < 5)).astype(int)
    
    df['time_of_day'] = pd.cut(df['hour'], bins=[-1, 6, 12, 18, 24], labels=[0, 1, 2, 3]).astype(int) # Night, Morn, Aft, Eve
    df['season'] = (df['month'] % 12 // 3).astype(int)
    df['week_of_year'] = df['time'].dt.isocalendar().week.astype(int)
    df['is_month_start'] = df['time'].dt.is_month_start.astype(int)
    df['is_month_end'] = df['time'].dt.is_month_end.astype(int)
    epoch_start = pd.Timestamp('2019-01-01')
    df['days_since_epoch'] = (df['time'] - epoch_start).dt.days

    # Lags & Rolling (The Heavy Hitters)
    # Ensure no leakage across stations (groupby)
    # Since we sorted by station+time, we can use groupby().shift()
    logger.info("   -> Generating Lags...")
    g = df.groupby('station_name')['pickups']
    
    for lag in [1, 2, 3, 6, 12, 24, 48, 168]:
        df[f'lag_{lag}h'] = g.shift(lag)
        
    logger.info("   -> Generating Rolling Stats...")
    # Rolling is slower.
    for w in [4, 12, 24, 168]:
        # shift(1) to avoid leakage (rolling includes current row by default?)
        # rolling(closed='left') usage? Safer to shift first.
        # But rolling on 'pickups' includes current value? No, we want PAST rolling.
        # So we take shift(1) then rolling.
        shifted = g.shift(1)
        r = shifted.rolling(window=w, min_periods=1)
        df[f'roll_mean_{w}h'] = r.mean()
        df[f'roll_std_{w}h'] = r.std().fillna(0)
        df[f'roll_min_{w}h'] = r.min()
        df[f'roll_max_{w}h'] = r.max()
        
    # EMA
    df['ema_24h'] = g.transform(lambda x: x.ewm(span=24, adjust=False).mean().shift(1))

    # Clean NaNs (created by lags)
    df.dropna(inplace=True)
    logger.info(f"✅ Features Engineered. Final Rows: {len(df)}")

    # 4. Prepare X, y
    feature_cols = [
        "temp", "prcp", "wspd", "is_holiday", "day_of_week", "is_weekend", "days_to_holiday", "days_since_holiday", 
        "hour", "day", "month", "year", "quarter", "hour_sin", "hour_cos", "month_sin", "month_cos", 
        "day_sin", "day_cos", "is_morning_rush", "is_evening_rush", "is_rush_hour", "is_business_hours", 
        "time_of_day", "season", "week_of_year", "is_month_start", "is_month_end", "days_since_epoch", 
        "lag_1h", "lag_2h", "lag_3h", "lag_6h", "lag_12h", "lag_24h", "lag_48h", "lag_168h", 
        "roll_mean_4h", "roll_std_4h", "roll_min_4h", "roll_max_4h", 
        "roll_mean_12h", "roll_std_12h", "roll_min_12h", "roll_max_12h", 
        "roll_mean_24h", "roll_std_24h", "roll_min_24h", "roll_max_24h", 
        "roll_mean_168h", "roll_std_168h", "roll_min_168h", "roll_max_168h", 
        "ema_24h"
    ]
    
    # Ensure all cols exist
    for c in feature_cols:
        if c not in df.columns:
            logger.warning(f"Feature {c} missing! Filling with 0.")
            df[c] = 0

    X = df[feature_cols]
    y = df['pickups']
    
    # Retrain Scalers (Matched to this exact data)
    logger.info("🔄 Retraining Scalers (Matched)...")
    scaler_tree = StandardScaler()
    X_scaled = scaler_tree.fit_transform(X)
    joblib.dump(scaler_tree, os.path.join(MODELS_DIR, "scaler_tree_server.save")) # Use distinct name first
    
    scaler_y = StandardScaler()
    y_scaled = scaler_y.fit_transform(y.values.reshape(-1, 1))
    joblib.dump(scaler_y, os.path.join(MODELS_DIR, "scaler_y_server.save"))

    # 5. Train XGBoost
    logger.info("🏋️ Training XGBoost Model on Server...")
    model = xgb.XGBRegressor(
        objective='reg:squarederror',
        n_estimators=60, # Lightweight but effective
        learning_rate=0.1,
        max_depth=6,
        n_jobs=-1
    )
    
    model.fit(X_scaled, y_scaled)
    
    # Save
    save_path = os.path.join(MODELS_DIR, "xgb_server.json")
    model.save_model(save_path)
    logger.info(f"✅ Model saved to {save_path}")
    
    logger.info("🎉 Training Complete.")

if __name__ == "__main__":
    train_server_model()
