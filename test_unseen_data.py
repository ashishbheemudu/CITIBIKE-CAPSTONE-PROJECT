#!/usr/bin/env python3
"""
Test Trained Model on Unseen 2025 Data (Sept, Oct, Nov)
========================================================
This script:
1. Loads and merges raw trip CSV files from the 'new months' folder
2. Aggregates to hourly demand per station
3. Engineers all 56 features (matching training pipeline)
4. Loads the trained ensemble models
5. Generates predictions and evaluates performance
"""

import pandas as pd
import numpy as np
import json
import os
import glob
import joblib
import warnings
from datetime import datetime, timedelta
import holidays

warnings.filterwarnings('ignore')

# ==============================================================================
# CONFIGURATION
# ==============================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
NEW_DATA_DIR = os.path.join(BASE_DIR, "new months")
MODELS_DIR = os.path.join(BASE_DIR, "backend", "models")
OUTPUT_DIR = os.path.join(BASE_DIR, "test_results_2025")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Feature names (must match training)
FEATURE_NAMES = [
    "temp", "prcp", "wspd", "is_holiday", "day_of_week", "is_weekend",
    "days_to_holiday", "days_since_holiday", "hour", "day", "month", "year", "quarter",
    "hour_sin", "hour_cos", "month_sin", "month_cos", "day_sin", "day_cos",
    "is_morning_rush", "is_evening_rush", "is_rush_hour", "is_business_hours",
    "time_of_day", "season", "week_of_year", "is_month_start", "is_month_end",
    "days_since_epoch",
    "lag_1h", "lag_2h", "lag_3h", "lag_6h", "lag_12h", "lag_24h", "lag_48h", "lag_168h",
    "roll_mean_4h", "roll_std_4h", "roll_min_4h", "roll_max_4h",
    "roll_mean_12h", "roll_std_12h", "roll_min_12h", "roll_max_12h",
    "roll_mean_24h", "roll_std_24h", "roll_min_24h", "roll_max_24h",
    "roll_mean_168h", "roll_std_168h", "roll_min_168h", "roll_max_168h",
    "ema_24h", "demand_change_1h", "demand_change_24h"
]

# ==============================================================================
# STEP 1: LOAD AND MERGE RAW DATA
# ==============================================================================
def load_raw_data():
    """Load all CSV files from new months folder"""
    print("=" * 60)
    print("STEP 1: LOADING RAW DATA")
    print("=" * 60)
    
    all_files = []
    
    # Sept 2025
    sept_files = glob.glob(os.path.join(NEW_DATA_DIR, "202509-citibike-tripdata", "*.csv"))
    print(f"Found {len(sept_files)} Sept 2025 files")
    all_files.extend(sept_files)
    
    # Oct 2025
    oct_files = glob.glob(os.path.join(NEW_DATA_DIR, "202510-citibike-tripdata", "*.csv"))
    print(f"Found {len(oct_files)} Oct 2025 files")
    all_files.extend(oct_files)
    
    # Nov 2025 (Jersey City - top level)
    nov_files = glob.glob(os.path.join(NEW_DATA_DIR, "JC-*.csv"))
    print(f"Found {len(nov_files)} Nov 2025 files")
    all_files.extend(nov_files)
    
    # Load and concat
    dfs = []
    for f in all_files:
        print(f"  Loading {os.path.basename(f)}...", end=" ")
        try:
            df = pd.read_csv(f)
            print(f"{len(df):,} rows")
            dfs.append(df)
        except Exception as e:
            print(f"ERROR: {e}")
    
    df_raw = pd.concat(dfs, ignore_index=True)
    print(f"\n✅ Total raw trips loaded: {len(df_raw):,}")
    print(f"   Columns: {list(df_raw.columns)}")
    
    return df_raw

# ==============================================================================
# STEP 2: AGGREGATE TO HOURLY DEMAND
# ==============================================================================
def aggregate_hourly(df_raw):
    """Aggregate trips to hourly net demand per station"""
    print("\n" + "=" * 60)
    print("STEP 2: AGGREGATING TO HOURLY DEMAND")
    print("=" * 60)
    
    # Parse timestamps
    df_raw['started_at'] = pd.to_datetime(df_raw['started_at'])
    df_raw['ended_at'] = pd.to_datetime(df_raw['ended_at'])
    
    # Floor to hour
    df_raw['start_h'] = df_raw['started_at'].dt.floor('h')
    df_raw['end_h'] = df_raw['ended_at'].dt.floor('h')
    
    # Outbound (departures = we'll track as pickups)
    outbound = df_raw.groupby(['start_station_name', 'start_h']).size().reset_index(name='pickups')
    outbound.columns = ['station_name', 'time', 'pickups']
    
    # Inbound (arrivals = dropoffs)
    inbound = df_raw.groupby(['end_station_name', 'end_h']).size().reset_index(name='dropoffs')
    inbound.columns = ['station_name', 'time', 'dropoffs']
    
    # Merge
    df_agg = outbound.merge(inbound, on=['station_name', 'time'], how='outer').fillna(0)
    df_agg['pickups'] = df_agg['pickups'].astype(int)
    df_agg['dropoffs'] = df_agg['dropoffs'].astype(int)
    df_agg['net_demand'] = df_agg['dropoffs'] - df_agg['pickups']
    
    # Remove empty station names
    df_agg = df_agg[df_agg['station_name'].notna() & (df_agg['station_name'] != '')]
    
    print(f"✅ Aggregated to {len(df_agg):,} hourly records")
    print(f"   Unique stations: {df_agg['station_name'].nunique()}")
    print(f"   Date range: {df_agg['time'].min()} to {df_agg['time'].max()}")
    
    return df_agg

# ==============================================================================
# STEP 3: FEATURE ENGINEERING
# ==============================================================================
def engineer_features(df_agg):
    """Create all 56 features matching the training pipeline"""
    print("\n" + "=" * 60)
    print("STEP 3: FEATURE ENGINEERING (56 Features)")
    print("=" * 60)
    
    df = df_agg.copy()
    df = df.sort_values(['station_name', 'time'])
    
    # --- WEATHER (Synthetic for now, can be replaced with real) ---
    print("  Creating weather features...")
    days_since_jan1 = (df['time'] - pd.Timestamp('2025-01-01')).dt.total_seconds() / 86400
    # Seasonal temperature pattern
    df['temp'] = 15 + 12 * np.sin(2 * np.pi * (days_since_jan1 - 80) / 365)  # Peak in summer
    df['temp'] += np.random.normal(0, 3, len(df))  # Add noise
    df['prcp'] = np.where(np.random.random(len(df)) < 0.1, np.random.exponential(1, len(df)), 0)
    df['wspd'] = 8 + 4 * np.sin(2 * np.pi * days_since_jan1 / 365) + np.random.normal(0, 2, len(df))
    df['wspd'] = df['wspd'].clip(0, 30)
    
    # --- HOLIDAYS ---
    print("  Creating holiday features...")
    us_holidays = holidays.US(years=[2025])
    df['is_holiday'] = df['time'].dt.date.apply(lambda x: 1 if x in us_holidays else 0)
    
    # Day of week & weekend
    df['day_of_week'] = df['time'].dt.dayofweek
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    
    # Holiday proximity
    holiday_dates = sorted(us_holidays.keys())
    def days_to_next_holiday(date):
        future = [h for h in holiday_dates if h > date.date()]
        return (min(future) - date.date()).days if future else 30
    def days_since_last_holiday(date):
        past = [h for h in holiday_dates if h < date.date()]
        return (date.date() - max(past)).days if past else 30
    
    df['days_to_holiday'] = df['time'].apply(days_to_next_holiday)
    df['days_since_holiday'] = df['time'].apply(days_since_last_holiday)
    
    # --- TEMPORAL FEATURES ---
    print("  Creating temporal features...")
    df['hour'] = df['time'].dt.hour
    df['day'] = df['time'].dt.day
    df['month'] = df['time'].dt.month
    df['year'] = df['time'].dt.year
    df['quarter'] = (df['month'] - 1) // 3 + 1
    
    # Cyclical encoding
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    df['day_sin'] = np.sin(2 * np.pi * df['day'] / 31)
    df['day_cos'] = np.cos(2 * np.pi * df['day'] / 31)
    
    # Rush hours
    df['is_morning_rush'] = ((df['hour'] >= 7) & (df['hour'] <= 9)).astype(int)
    df['is_evening_rush'] = ((df['hour'] >= 17) & (df['hour'] <= 19)).astype(int)
    df['is_rush_hour'] = (df['is_morning_rush'] | df['is_evening_rush']).astype(int)
    df['is_business_hours'] = ((df['hour'] >= 9) & (df['hour'] <= 17) & (df['day_of_week'] < 5)).astype(int)
    
    # Time of day (1=morning, 2=afternoon, 3=evening, 4=night)
    conditions = [
        (df['hour'] >= 5) & (df['hour'] < 12),   # Morning
        (df['hour'] >= 12) & (df['hour'] < 17),  # Afternoon
        (df['hour'] >= 17) & (df['hour'] < 21),  # Evening
    ]
    choices = [1, 2, 3]
    df['time_of_day'] = np.select(conditions, choices, default=4)  # Night
    
    # Season
    def get_season(month):
        if month in [12, 1, 2]: return 1
        elif month in [3, 4, 5]: return 2
        elif month in [6, 7, 8]: return 3
        else: return 4
    df['season'] = df['month'].apply(get_season)
    
    df['week_of_year'] = df['time'].dt.isocalendar().week.astype(int)
    df['is_month_start'] = (df['day'] <= 3).astype(int)
    df['is_month_end'] = (df['day'] >= 28).astype(int)
    
    # Days since epoch
    epoch = pd.Timestamp('2020-01-01')
    df['days_since_epoch'] = (df['time'] - epoch).dt.total_seconds() / 86400
    
    # --- LAG FEATURES ---
    print("  Creating lag features (this may take a while)...")
    grouped = df.groupby('station_name')['pickups']
    
    lags = [1, 2, 3, 6, 12, 24, 48, 168]
    for lag in lags:
        df[f'lag_{lag}h'] = grouped.shift(lag).fillna(0)
    
    # --- ROLLING FEATURES ---
    print("  Creating rolling window features...")
    windows = [4, 12, 24, 168]
    for window in windows:
        df[f'roll_mean_{window}h'] = grouped.transform(lambda x: x.shift(1).rolling(window, min_periods=1).mean()).fillna(0)
        df[f'roll_std_{window}h'] = grouped.transform(lambda x: x.shift(1).rolling(window, min_periods=1).std()).fillna(1)
        df[f'roll_min_{window}h'] = grouped.transform(lambda x: x.shift(1).rolling(window, min_periods=1).min()).fillna(0)
        df[f'roll_max_{window}h'] = grouped.transform(lambda x: x.shift(1).rolling(window, min_periods=1).max()).fillna(0)
    
    # EMA
    df['ema_24h'] = grouped.transform(lambda x: x.shift(1).ewm(span=24, adjust=False).mean()).fillna(0)
    
    # Change features
    df['demand_change_1h'] = (df['pickups'] - df['lag_1h']).fillna(0)
    df['demand_change_24h'] = (df['pickups'] - df['lag_24h']).fillna(0)
    
    print(f"✅ Feature engineering complete!")
    print(f"   Total columns: {len(df.columns)}")
    
    return df

# ==============================================================================
# STEP 4: LOAD MODELS
# ==============================================================================
def load_models():
    """Load trained ensemble models"""
    print("\n" + "=" * 60)
    print("STEP 4: LOADING TRAINED MODELS")
    print("=" * 60)
    
    models = {}
    
    # XGBoost
    xgb_path = os.path.join(MODELS_DIR, "xgb.json")
    if os.path.exists(xgb_path):
        import xgboost as xgb
        models['xgb'] = xgb.XGBRegressor()
        models['xgb'].load_model(xgb_path)
        print(f"✅ Loaded XGBoost from {xgb_path}")
    
    # LightGBM
    lgb_path = os.path.join(MODELS_DIR, "lgb.pkl")
    if os.path.exists(lgb_path):
        models['lgb'] = joblib.load(lgb_path)
        print(f"✅ Loaded LightGBM from {lgb_path}")
    
    # CatBoost
    cb_path = os.path.join(MODELS_DIR, "cb.cbm")
    if os.path.exists(cb_path):
        from catboost import CatBoostRegressor
        models['cb'] = CatBoostRegressor()
        models['cb'].load_model(cb_path)
        print(f"✅ Loaded CatBoost from {cb_path}")
    
    # Scalers
    scaler_tree = joblib.load(os.path.join(MODELS_DIR, "scaler_tree.save"))
    scaler_y = joblib.load(os.path.join(MODELS_DIR, "scaler_y.save"))
    print(f"✅ Loaded scalers")
    
    return models, scaler_tree, scaler_y

# ==============================================================================
# STEP 5: PREDICT AND EVALUATE
# ==============================================================================
def predict_and_evaluate(df, models, scaler_tree, scaler_y):
    """Run predictions and calculate metrics"""
    print("\n" + "=" * 60)
    print("STEP 5: PREDICTION AND EVALUATION")
    print("=" * 60)
    
    # Prepare features
    X = df[FEATURE_NAMES].fillna(0)
    y_true = df['pickups'].values
    
    print(f"Test set size: {len(X):,}")
    
    # Scale features
    X_scaled = scaler_tree.transform(X)
    
    # Get predictions from each model
    predictions = {}
    ensemble_weights = {'xgb': 0.4, 'lgb': 0.3, 'cb': 0.3}  # Slightly favor XGB
    
    for name, model in models.items():
        pred = model.predict(X_scaled)
        predictions[name] = pred
        print(f"  {name}: Generated {len(pred):,} predictions")
    
    # Ensemble (weighted average)
    final_pred_scaled = np.zeros(len(X))
    total_weight = 0
    for name, pred in predictions.items():
        weight = ensemble_weights.get(name, 0.33)
        final_pred_scaled += weight * pred
        total_weight += weight
    final_pred_scaled /= total_weight
    
    # Inverse transform
    final_pred = scaler_y.inverse_transform(final_pred_scaled.reshape(-1, 1)).flatten()
    
    # Ensure non-negative (pickups can't be negative)
    final_pred = np.maximum(final_pred, 0)
    
    # ==== EVALUATION METRICS ====
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    
    mae = mean_absolute_error(y_true, final_pred)
    rmse = np.sqrt(mean_squared_error(y_true, final_pred))
    r2 = r2_score(y_true, final_pred)
    
    # MAPE (avoid division by zero)
    mask = y_true > 0
    mape = np.mean(np.abs((y_true[mask] - final_pred[mask]) / y_true[mask])) * 100 if mask.sum() > 0 else 0
    
    print("\n" + "=" * 60)
    print("📊 EVALUATION RESULTS")
    print("=" * 60)
    print(f"  MAE:   {mae:.2f} bikes/hour")
    print(f"  RMSE:  {rmse:.2f} bikes/hour")
    print(f"  R²:    {r2:.4f}")
    print(f"  MAPE:  {mape:.2f}%")
    
    # Save results
    results = {
        'mae': mae,
        'rmse': rmse,
        'r2': r2,
        'mape': mape,
        'test_size': len(X),
        'date_range': f"{df['time'].min()} to {df['time'].max()}"
    }
    
    with open(os.path.join(OUTPUT_DIR, "evaluation_metrics.json"), 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n✅ Saved metrics to {OUTPUT_DIR}/evaluation_metrics.json")
    
    # Save predictions
    df['predicted'] = final_pred
    df[['station_name', 'time', 'pickups', 'predicted']].to_csv(
        os.path.join(OUTPUT_DIR, "predictions.csv"), index=False
    )
    print(f"✅ Saved predictions to {OUTPUT_DIR}/predictions.csv")
    
    # Plot
    try:
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # 1. Actual vs Predicted scatter
        ax1 = axes[0, 0]
        sample_idx = np.random.choice(len(y_true), min(5000, len(y_true)), replace=False)
        ax1.scatter(y_true[sample_idx], final_pred[sample_idx], alpha=0.3, s=5)
        ax1.plot([0, y_true.max()], [0, y_true.max()], 'r--', lw=2, label='Perfect')
        ax1.set_xlabel('Actual Pickups')
        ax1.set_ylabel('Predicted Pickups')
        ax1.set_title(f'Actual vs Predicted (R²={r2:.3f})')
        ax1.legend()
        
        # 2. Error distribution
        ax2 = axes[0, 1]
        errors = final_pred - y_true
        ax2.hist(errors, bins=50, edgecolor='black', alpha=0.7)
        ax2.axvline(0, color='red', linestyle='--')
        ax2.set_xlabel('Prediction Error')
        ax2.set_ylabel('Frequency')
        ax2.set_title(f'Error Distribution (MAE={mae:.2f})')
        
        # 3. Hourly pattern
        ax3 = axes[1, 0]
        hourly = df.groupby('hour')[['pickups', 'predicted']].mean()
        hourly.plot(ax=ax3, marker='o')
        ax3.set_xlabel('Hour of Day')
        ax3.set_ylabel('Average Pickups')
        ax3.set_title('Hourly Pattern: Actual vs Predicted')
        ax3.legend(['Actual', 'Predicted'])
        
        # 4. Daily trend
        ax4 = axes[1, 1]
        df['date'] = df['time'].dt.date
        daily = df.groupby('date')[['pickups', 'predicted']].sum()
        daily.plot(ax=ax4)
        ax4.set_xlabel('Date')
        ax4.set_ylabel('Total Pickups')
        ax4.set_title('Daily Trend: Actual vs Predicted')
        ax4.legend(['Actual', 'Predicted'])
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, "evaluation_plots.png"), dpi=150, bbox_inches='tight')
        print(f"✅ Saved plots to {OUTPUT_DIR}/evaluation_plots.png")
        plt.close()
        
    except Exception as e:
        print(f"⚠️ Could not generate plots: {e}")
    
    return results

# ==============================================================================
# MAIN
# ==============================================================================
if __name__ == "__main__":
    print("\n" + "🚴" * 30)
    print("  CITIBIKE MODEL EVALUATION ON UNSEEN 2025 DATA")
    print("🚴" * 30 + "\n")
    
    # Execute pipeline
    df_raw = load_raw_data()
    df_agg = aggregate_hourly(df_raw)
    df_features = engineer_features(df_agg)
    models, scaler_tree, scaler_y = load_models()
    results = predict_and_evaluate(df_features, models, scaler_tree, scaler_y)
    
    print("\n" + "=" * 60)
    print("🎉 EVALUATION COMPLETE!")
    print("=" * 60)
    print(f"\nResults saved to: {OUTPUT_DIR}/")
    print("  - evaluation_metrics.json")
    print("  - predictions.csv")
    print("  - evaluation_plots.png")
