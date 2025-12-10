"""
CSV Evaluator Service
=====================
Processes uploaded CitiBike trip CSV files, engineers features,
runs ensemble predictions, and returns evaluation metrics.
"""

import pandas as pd
import numpy as np
import json
import os
import joblib
import holidays
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("CSVEvaluator")

# Constants
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")

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


class CSVEvaluator:
    def __init__(self):
        self.models = {}
        self.scaler_tree = None
        self.scaler_y = None
        self._load_models()
    
    def _load_models(self):
        """Load trained models and scalers"""
        try:
            import xgboost as xgb
            import lightgbm as lgb
            from catboost import CatBoostRegressor
            
            # XGBoost
            xgb_path = os.path.join(MODELS_DIR, "xgb.json")
            if os.path.exists(xgb_path):
                self.models['xgb'] = xgb.XGBRegressor()
                self.models['xgb'].load_model(xgb_path)
                logger.info("✅ Loaded XGBoost")
            
            # LightGBM
            lgb_path = os.path.join(MODELS_DIR, "lgb.pkl")
            if os.path.exists(lgb_path):
                self.models['lgb'] = joblib.load(lgb_path)
                logger.info("✅ Loaded LightGBM")
            
            # CatBoost
            cb_path = os.path.join(MODELS_DIR, "cb.cbm")
            if os.path.exists(cb_path):
                self.models['cb'] = CatBoostRegressor()
                self.models['cb'].load_model(cb_path)
                logger.info("✅ Loaded CatBoost")
            
            # Scalers
            self.scaler_tree = joblib.load(os.path.join(MODELS_DIR, "scaler_tree.save"))
            self.scaler_y = joblib.load(os.path.join(MODELS_DIR, "scaler_y.save"))
            logger.info("✅ Loaded scalers")
            
        except Exception as e:
            logger.error(f"❌ Error loading models: {e}")
    
    def process_csv(self, csv_content: bytes) -> dict:
        """
        Process uploaded CSV and return evaluation results.
        
        Args:
            csv_content: Raw bytes of uploaded CSV file
            
        Returns:
            dict with metrics, sample_predictions, hourly_pattern, etc.
        """
        try:
            # 1. Parse CSV
            from io import BytesIO
            df_raw = pd.read_csv(BytesIO(csv_content))
            logger.info(f"📊 Loaded CSV: {len(df_raw):,} rows")
            
            # Validate columns
            required_cols = ['started_at', 'start_station_name']
            missing = [c for c in required_cols if c not in df_raw.columns]
            if missing:
                return {"error": f"Missing required columns: {missing}"}
            
            # 2. Aggregate to hourly
            df_agg = self._aggregate_hourly(df_raw)
            logger.info(f"📊 Aggregated: {len(df_agg):,} hourly records")
            
            # 3. Engineer features
            df = self._engineer_features(df_agg)
            logger.info(f"📊 Features: {len(df.columns)} columns")
            
            # 4. Predict and evaluate
            results = self._predict_and_evaluate(df)
            
            return results
            
        except Exception as e:
            logger.error(f"❌ Error processing CSV: {e}")
            import traceback
            traceback.print_exc()
            return {"error": str(e)}
    
    def _aggregate_hourly(self, df_raw):
        """Aggregate trips to hourly demand per station"""
        df_raw['started_at'] = pd.to_datetime(df_raw['started_at'])
        df_raw['start_h'] = df_raw['started_at'].dt.floor('h')
        
        # Count pickups
        df_agg = df_raw.groupby(['start_station_name', 'start_h']).size().reset_index(name='pickups')
        df_agg.columns = ['station_name', 'time', 'pickups']
        
        # Remove empty stations
        df_agg = df_agg[df_agg['station_name'].notna() & (df_agg['station_name'] != '')]
        
        return df_agg
    
    def _engineer_features(self, df_agg):
        """Create all 56 features"""
        df = df_agg.copy()
        df = df.sort_values(['station_name', 'time'])
        
        # Weather (synthetic)
        days_since_jan1 = (df['time'] - pd.Timestamp('2025-01-01')).dt.total_seconds() / 86400
        df['temp'] = 15 + 12 * np.sin(2 * np.pi * (days_since_jan1 - 80) / 365)
        df['prcp'] = 0.0
        df['wspd'] = 8.0
        
        # Holidays
        years = df['time'].dt.year.unique().tolist()
        us_holidays = holidays.US(years=years)
        df['is_holiday'] = df['time'].dt.date.apply(lambda x: 1 if x in us_holidays else 0)
        
        # Day of week
        df['day_of_week'] = df['time'].dt.dayofweek
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        
        # Holiday proximity (simplified)
        df['days_to_holiday'] = 7
        df['days_since_holiday'] = 7
        
        # Temporal
        df['hour'] = df['time'].dt.hour
        df['day'] = df['time'].dt.day
        df['month'] = df['time'].dt.month
        df['year'] = df['time'].dt.year
        df['quarter'] = (df['month'] - 1) // 3 + 1
        
        # Cyclical
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
        
        # Time of day
        conditions = [
            (df['hour'] >= 5) & (df['hour'] < 12),
            (df['hour'] >= 12) & (df['hour'] < 17),
            (df['hour'] >= 17) & (df['hour'] < 21),
        ]
        df['time_of_day'] = np.select(conditions, [1, 2, 3], default=4)
        
        # Season
        def get_season(month):
            if month in [12, 1, 2]: return 1
            elif month in [3, 4, 5]: return 2
            elif month in [6, 7, 8]: return 3
            return 4
        df['season'] = df['month'].apply(get_season)
        
        df['week_of_year'] = df['time'].dt.isocalendar().week.astype(int)
        df['is_month_start'] = (df['day'] <= 3).astype(int)
        df['is_month_end'] = (df['day'] >= 28).astype(int)
        
        epoch = pd.Timestamp('2020-01-01')
        df['days_since_epoch'] = (df['time'] - epoch).dt.total_seconds() / 86400
        
        # Lag features
        grouped = df.groupby('station_name')['pickups']
        for lag in [1, 2, 3, 6, 12, 24, 48, 168]:
            df[f'lag_{lag}h'] = grouped.shift(lag).fillna(0)
        
        # Rolling features
        for window in [4, 12, 24, 168]:
            df[f'roll_mean_{window}h'] = grouped.transform(lambda x: x.shift(1).rolling(window, min_periods=1).mean()).fillna(0)
            df[f'roll_std_{window}h'] = grouped.transform(lambda x: x.shift(1).rolling(window, min_periods=1).std()).fillna(1)
            df[f'roll_min_{window}h'] = grouped.transform(lambda x: x.shift(1).rolling(window, min_periods=1).min()).fillna(0)
            df[f'roll_max_{window}h'] = grouped.transform(lambda x: x.shift(1).rolling(window, min_periods=1).max()).fillna(0)
        
        df['ema_24h'] = grouped.transform(lambda x: x.shift(1).ewm(span=24, adjust=False).mean()).fillna(0)
        df['demand_change_1h'] = (df['pickups'] - df['lag_1h']).fillna(0)
        df['demand_change_24h'] = (df['pickups'] - df['lag_24h']).fillna(0)
        
        return df
    
    def _predict_and_evaluate(self, df):
        """Run ensemble predictions and calculate metrics"""
        X = df[FEATURE_NAMES].fillna(0)
        y_true = df['pickups'].values
        
        # Scale
        X_scaled = self.scaler_tree.transform(X)
        
        # Ensemble prediction
        ensemble_weights = {'xgb': 0.4, 'lgb': 0.3, 'cb': 0.3}
        final_pred_scaled = np.zeros(len(X))
        total_weight = 0
        
        for name, model in self.models.items():
            pred = model.predict(X_scaled)
            weight = ensemble_weights.get(name, 0.33)
            final_pred_scaled += weight * pred
            total_weight += weight
        
        final_pred_scaled /= total_weight
        final_pred = self.scaler_y.inverse_transform(final_pred_scaled.reshape(-1, 1)).flatten()
        final_pred = np.maximum(final_pred, 0)
        
        # Metrics
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
        
        mae = float(mean_absolute_error(y_true, final_pred))
        rmse = float(np.sqrt(mean_squared_error(y_true, final_pred)))
        r2 = float(r2_score(y_true, final_pred))
        
        mask = y_true > 0
        mape = float(np.mean(np.abs((y_true[mask] - final_pred[mask]) / y_true[mask])) * 100) if mask.sum() > 0 else 0
        
        # Hourly pattern
        df['predicted'] = final_pred
        hourly_pattern = df.groupby('hour').agg({
            'pickups': 'mean',
            'predicted': 'mean'
        }).reset_index().to_dict('records')
        
        # Sample predictions (top 10 stations)
        top_stations = df.groupby('station_name')['pickups'].sum().nlargest(10).index.tolist()
        sample_df = df[df['station_name'].isin(top_stations)][['station_name', 'time', 'pickups', 'predicted']].head(100)
        sample_predictions = sample_df.to_dict('records')
        
        # Date range
        date_range = f"{df['time'].min()} to {df['time'].max()}"
        
        return {
            "success": True,
            "metrics": {
                "mae": round(mae, 2),
                "rmse": round(rmse, 2),
                "r2": round(r2, 4),
                "mape": round(mape, 2)
            },
            "summary": {
                "total_trips": int(df['pickups'].sum()),
                "total_records": len(df),
                "unique_stations": df['station_name'].nunique(),
                "date_range": date_range
            },
            "hourly_pattern": hourly_pattern,
            "sample_predictions": sample_predictions
        }


# Global instance
csv_evaluator = CSVEvaluator()
