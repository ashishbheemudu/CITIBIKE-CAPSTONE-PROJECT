import pandas as pd
import numpy as np
import json
import os
import joblib
import logging
from datetime import datetime, timedelta

# Init Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("PredictionService")

# Constants
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")

class PredictionService:
    def __init__(self, reference_data=None, historical_data=None):
        self.models = {}
        self.scalers = {}
        self.reference_data = reference_data
        self.historical_data = historical_data
        self.feature_names = []
        self.ensemble_weights = {}
    # REMOVED _load_models from __init__ to enable LAZY LOADING
    # self._load_models() 
    
    def _lazy_load_models(self):
        """Load models ONLY when needed (Lazy Loading) to prevent startup OOM"""
        if self.models: return # Already loaded

        try:
            logger.info("🚀 Lazy Loading ML models...")

            # 1. Load feature names
            feature_path = os.path.join(MODELS_DIR, "feature_names.json")
            if os.path.exists(feature_path):
                with open(feature_path, 'r') as f:
                    self.feature_names = json.load(f)
                logger.info(f"✅ Features: {len(self.feature_names)} features loaded")

            # 2. Load ensemble config
            ensemble_path = os.path.join(MODELS_DIR, "ensemble_config.json")
            if os.path.exists(ensemble_path):
                with open(ensemble_path, 'r') as f:
                    config = json.load(f)
                    self.ensemble_weights = config.get('weights', {})
                logger.info(f"✅ Ensemble weights: {self.ensemble_weights}")

            # 3. Load models (Import here to save memory until needed)
            import xgboost as xgb
            import lightgbm as lgb
            from catboost import CatBoostRegressor

            # XGBoost (101MB)
            xgb_path = os.path.join(MODELS_DIR, "xgb.json")
            if os.path.exists(xgb_path):
                self.models['xgb'] = xgb.XGBRegressor()
                self.models['xgb'].load_model(xgb_path)
                logger.info("✅ Loaded XGBoost")

            # LightGBM (3MB)
            lgb_path = os.path.join(MODELS_DIR, "lgb.pkl")
            if os.path.exists(lgb_path):
                self.models['lgb'] = joblib.load(lgb_path)
                logger.info("✅ Loaded LightGBM")

            # CatBoost (16MB)
            cb_path = os.path.join(MODELS_DIR, "cb.cbm")
            if os.path.exists(cb_path):
                self.models['cb'] = CatBoostRegressor()
                self.models['cb'].load_model(cb_path)
                logger.info("✅ Loaded CatBoost")

            # 4. Load scalers
            scaler_tree_path = os.path.join(MODELS_DIR, "scaler_tree.save")
            if os.path.exists(scaler_tree_path):
                self.scalers['tree'] = joblib.load(scaler_tree_path)
                logger.info(f"✅ Loaded feature scaler")

            scaler_y_path = os.path.join(MODELS_DIR, "scaler_y.save")
            if os.path.exists(scaler_y_path):
                self.scalers['y'] = joblib.load(scaler_y_path)
                logger.info("✅ Loaded target scaler")
            
            logger.info(f"🎉 Lazy Loaded {len(self.models)} models!")
        except Exception as e:
            logger.error(f"❌ Error lazy loading models: {e}")
            import traceback
            traceback.print_exc()

    def _lazy_load_data(self):
        """Load historical data ONLY when needed (Lazy Loading) with aggressive pruning"""
        if self.historical_data is not None: return # Already loaded

        try:
            logger.info("READING DATA: Lazy loading historical data...")
            hist_path = os.path.join(BASE_DIR, "data", "v1_core", "final_station_demand_robust_features.parquet")
            
            if os.path.exists(hist_path):
                # MEMORY OPTIMIZATION: Only load essential columns
                # BEAST MODE UPDATE: AWS has 1GB RAM, can load 'wspd' and other columns if needed.
                self.historical_data = pd.read_parquet(
                    hist_path, 
                    columns=['time', 'station_name', 'pickups', 'temp', 'prcp', 'wspd']
                )
                
                # Optimize types
                for col in self.historical_data.select_dtypes(include=['float64']).columns:
                    self.historical_data[col] = self.historical_data[col].astype('float32')
                
                if 'time' in self.historical_data.columns:
                    self.historical_data['time'] = pd.to_datetime(self.historical_data['time'])
                
                logger.info(f"✅ Lazy Loaded {len(self.historical_data)} rows of historical data (Optimized)")
            else:
                logger.warning("⚠️ Historical data file not found during lazy load")
                
        except Exception as e:
            logger.error(f"❌ Error lazy loading data: {e}")
            self.historical_data = pd.DataFrame() # Prevent NoneType errors

    def predict(self, station_name, start_time, hours_ahead=48):
        """Generate predictions using REAL ML models"""
        try:
            # Enforce Lazy Loading on first request
            self._lazy_load_models()
            self._lazy_load_data()
            
            if not self.models:
                logger.error("❌ No models loaded")
                raise ValueError("No models loaded - prediction service not initialized properly")

            # Create prediction timestamps
            start_dt = pd.to_datetime(start_time)
            timestamps = [start_dt + timedelta(hours=i) for i in range(hours_ahead)]
            
            # PRE-FILTER station data ONCE (major optimization - avoids 48x filtering)
            import time as time_module
            perf_start = time_module.time()
            station_cache = None
            if hasattr(self, 'historical_data') and self.historical_data is not None:
                ts_start = pd.Timestamp(start_dt).tz_localize(None) if pd.Timestamp(start_dt).tz else pd.Timestamp(start_dt)
                hist_station = self.historical_data[self.historical_data['station_name'] == station_name].copy()
                if not hist_station.empty:
                    hist_station['time'] = pd.to_datetime(hist_station['time']).dt.tz_localize(None)
                    before_ts = hist_station[hist_station['time'] < ts_start]
                    if not before_ts.empty:
                        station_cache = before_ts.tail(168)  # Last week
            logger.info(f"⏱️ Station filter: {(time_module.time() - perf_start)*1000:.0f}ms")

            # Create features for each hour (pass cached station data)
            features_list = []
            for ts in timestamps:
                features = self._create_features(station_name, ts, station_cache)
                features_list.append(features)

            X = pd.DataFrame(features_list, columns=self.feature_names)

            # Scale features
            if 'tree' in self.scalers:
                X_scaled = self.scalers['tree'].transform(X)
            else:
                X_scaled = X.values

            # Get predictions from each model
            predictions = {}
            for model_name, model in self.models.items():
                try:
                    pred_scaled = model.predict(X_scaled)
                    predictions[model_name] = pred_scaled
                    logger.info(f"✅ {model_name} predicted {len(pred_scaled)} values")
                except Exception as e:
                    logger.warning(f"⚠️ {model_name} prediction failed: {e}")

            # Ensemble predictions (weighted average)
            if predictions:
                final_pred_scaled = np.zeros(len(timestamps))
                total_weight = 0

                for model_name, pred in predictions.items():
                    weight = self.ensemble_weights.get(model_name, 1.0 / len(predictions))
                    final_pred_scaled += weight * pred
                    total_weight += weight

                if total_weight > 0:
                    final_pred_scaled = final_pred_scaled / total_weight

                # Inverse transform
                if 'y' in self.scalers:
                    final_pred = self.scalers['y'].inverse_transform(final_pred_scaled.reshape(-1, 1)).flatten()
                else:
                    final_pred = final_pred_scaled

                # Format output
                results = []
                for ts, pred_value in zip(timestamps, final_pred):
                    results.append({
                        'date': ts.isoformat(),
                        'predicted': float(pred_value)  # Allow negative for net_demand
                    })

                logger.info(f"🎯 Generated {len(results)} predictions")
                return results
            else:
                logger.error("❌ No predictions generated from any model")
                raise ValueError("No predictions generated - all models failed")

        except ValueError:
            raise
        except Exception as e:
            logger.error(f"❌ Prediction error: {e}")
            import traceback
            traceback.print_exc()
            raise ValueError(f"Prediction failed: {str(e)}")

    def _create_features(self, station_name, timestamp, station_cache=None):
        """Create all 56 features for a single prediction point"""
        
        # Use pre-cached station data if provided (MAJOR SPEEDUP)
        station_data = station_cache
        ts = pd.Timestamp(timestamp)
        if ts.tz is not None:
            ts = ts.tz_localize(None)
        
        # Only filter if no cache provided (backward compatibility)
        if station_cache is None:
            # Try historical data (Optimized Load)
            if hasattr(self, 'historical_data') and self.historical_data is not None:
                 # Already loaded (e.g. dev mode)
                 pass
            else:
                 # Load ONLY this station's data from disk
                 hist_path = os.path.join(BASE_DIR, "data", "v1_core", "final_station_demand_robust_features.parquet")
                 if os.path.exists(hist_path):
                     try:
                         # Use filters to read only specific station rows (Requires pyarrow)
                         self.historical_data = pd.read_parquet(
                             hist_path,
                             columns=['time', 'station_name', 'pickups', 'temp', 'prcp', 'wspd'],
                             filters=[('station_name', '==', station_name)]
                         )
                         self.historical_data['time'] = pd.to_datetime(self.historical_data['time'])
                         # Type optim
                         for col in self.historical_data.select_dtypes(include=['float64']).columns:
                             self.historical_data[col] = self.historical_data[col].astype('float32')
                     except Exception as e:
                         logger.warning(f"⚠️ Failed to filter-load parquet: {e}")

            # Filter for memory buffer
            if hasattr(self, 'historical_data') and self.historical_data is not None:
                hist_station = self.historical_data[(self.historical_data['station_name'] == station_name)].copy()
                if not hist_station.empty:
                    # Get data BEFORE the prediction time (for lag features)
                    hist_station['time'] = pd.to_datetime(hist_station['time']).dt.tz_localize(None)
                    before_ts = hist_station[hist_station['time'] < ts]
                    if not before_ts.empty:
                        station_data = before_ts.tail(168)  # Last week of data before prediction
        
            # Fallback to reference data if no historical data
            if station_data is None or (hasattr(station_data, 'empty') and station_data.empty):
                if self.reference_data is not None:
                    station_data = self.reference_data[
                        self.reference_data['station_name'] == station_name
                    ].tail(168)
        
        # ═══════════════════════════════════════════════════════════════
        # WEATHER FEATURES (3)
        # ═══════════════════════════════════════════════════════════════
        temp = 15.0
        prcp = 0.0
        wspd = 5.0
        
        if station_data is not None and not station_data.empty:
            if 'temp' in station_data.columns:
                val = station_data['temp'].iloc[-1]
                if not pd.isna(val): temp = val
            if 'prcp' in station_data.columns:
                val = station_data['prcp'].iloc[-1]
                if not pd.isna(val): prcp = val
            if 'wspd' in station_data.columns:
                val = station_data['wspd'].iloc[-1]
                if not pd.isna(val): wspd = val
        
        # ═══════════════════════════════════════════════════════════════
        # TEMPORAL FEATURES
        # ═══════════════════════════════════════════════════════════════
        hour = timestamp.hour
        day = timestamp.day
        month = timestamp.month
        year = timestamp.year
        quarter = (month - 1) // 3 + 1
        day_of_week = timestamp.weekday()
        week_of_year = timestamp.isocalendar()[1]
        
        # Cyclical encoding
        hour_sin = np.sin(2 * np.pi * hour / 24)
        hour_cos = np.cos(2 * np.pi * hour / 24)
        month_sin = np.sin(2 * np.pi * month / 12)
        month_cos = np.cos(2 * np.pi * month / 12)
        day_sin = np.sin(2 * np.pi * day / 31)
        day_cos = np.cos(2 * np.pi * day / 31)
        
        # Weekend/Holiday
        is_weekend = 1 if day_of_week >= 5 else 0
        is_holiday = 0  # TODO: could add holiday calendar
        days_to_holiday = 7  # Default
        days_since_holiday = 7  # Default
        
        # Rush hour
        is_morning_rush = 1 if 7 <= hour <= 9 else 0
        is_evening_rush = 1 if 17 <= hour <= 19 else 0
        is_rush_hour = 1 if is_morning_rush or is_evening_rush else 0
        is_business_hours = 1 if (9 <= hour <= 17 and day_of_week < 5) else 0
        
        # Time of day (1=morning, 2=afternoon, 3=evening, 4=night)
        if 5 <= hour < 12:
            time_of_day = 1
        elif 12 <= hour < 17:
            time_of_day = 2
        elif 17 <= hour < 21:
            time_of_day = 3
        else:
            time_of_day = 4
        
        # Season (1=winter, 2=spring, 3=summer, 4=fall)
        if month in [12, 1, 2]:
            season = 1
        elif month in [3, 4, 5]:
            season = 2
        elif month in [6, 7, 8]:
            season = 3
        else:
            season = 4
        
        # Month boundaries
        is_month_start = 1 if day <= 3 else 0
        is_month_end = 1 if day >= 28 else 0
        
        # Days since epoch
        epoch = pd.Timestamp('2020-01-01')
        ts_naive = pd.Timestamp(timestamp).tz_localize(None) if hasattr(pd.Timestamp(timestamp), 'tz') and pd.Timestamp(timestamp).tz else pd.Timestamp(timestamp)
        days_since_epoch = (ts_naive - epoch).total_seconds() / 86400
        
        # ═══════════════════════════════════════════════════════════════
        # LAG FEATURES (8)
        # ═══════════════════════════════════════════════════════════════
        lag_1h = 0.0
        lag_2h = 0.0
        lag_3h = 0.0
        lag_6h = 0.0
        lag_12h = 0.0
        lag_24h = 0.0
        lag_48h = 0.0
        lag_168h = 0.0
        
        if station_data is not None and 'pickups' in station_data.columns:
            demands = station_data['pickups'].values
            if len(demands) >= 1: lag_1h = demands[-1]
            if len(demands) >= 2: lag_2h = demands[-2]
            if len(demands) >= 3: lag_3h = demands[-3]
            if len(demands) >= 6: lag_6h = demands[-6]
            if len(demands) >= 12: lag_12h = demands[-12]
            if len(demands) >= 24: lag_24h = demands[-24]
            if len(demands) >= 48: lag_48h = demands[-48]
            if len(demands) >= 168: lag_168h = demands[-168]
        
        # ═══════════════════════════════════════════════════════════════
        # ROLLING WINDOW FEATURES (16)
        # ═══════════════════════════════════════════════════════════════
        roll_mean_4h = 0.0
        roll_std_4h = 1.0
        roll_min_4h = 0.0
        roll_max_4h = 0.0
        roll_mean_12h = 0.0
        roll_std_12h = 1.0
        roll_min_12h = 0.0
        roll_max_12h = 0.0
        roll_mean_24h = 0.0
        roll_std_24h = 1.0
        roll_min_24h = 0.0
        roll_max_24h = 0.0
        roll_mean_168h = 0.0
        roll_std_168h = 1.0
        roll_min_168h = 0.0
        roll_max_168h = 0.0
        
        if station_data is not None and 'pickups' in station_data.columns:
            demands = station_data['pickups'].values
            
            if len(demands) >= 4:
                roll_mean_4h = np.mean(demands[-4:])
                roll_std_4h = np.std(demands[-4:]) if np.std(demands[-4:]) > 0 else 1.0
                roll_min_4h = np.min(demands[-4:])
                roll_max_4h = np.max(demands[-4:])
            
            if len(demands) >= 12:
                roll_mean_12h = np.mean(demands[-12:])
                roll_std_12h = np.std(demands[-12:]) if np.std(demands[-12:]) > 0 else 1.0
                roll_min_12h = np.min(demands[-12:])
                roll_max_12h = np.max(demands[-12:])
            
            if len(demands) >= 24:
                roll_mean_24h = np.mean(demands[-24:])
                roll_std_24h = np.std(demands[-24:]) if np.std(demands[-24:]) > 0 else 1.0
                roll_min_24h = np.min(demands[-24:])
                roll_max_24h = np.max(demands[-24:])
            
            if len(demands) >= 168:
                roll_mean_168h = np.mean(demands[-168:])
                roll_std_168h = np.std(demands[-168:]) if np.std(demands[-168:]) > 0 else 1.0
                roll_min_168h = np.min(demands[-168:])
                roll_max_168h = np.max(demands[-168:])
        
        # ═══════════════════════════════════════════════════════════════
        # EMA AND CHANGE FEATURES (3)
        # ═══════════════════════════════════════════════════════════════
        ema_24h = 0.0
        demand_change_1h = 0.0
        demand_change_24h = 0.0
        
        if station_data is not None and 'pickups' in station_data.columns:
            demands = station_data['pickups'].values
            if len(demands) >= 24:
                # Exponential moving average
                ema_24h = pd.Series(demands).ewm(span=24, adjust=False).mean().iloc[-1]
            if len(demands) >= 2:
                demand_change_1h = demands[-1] - demands[-2]
            if len(demands) >= 25:
                demand_change_24h = demands[-1] - demands[-24]
        
        # ═══════════════════════════════════════════════════════════════
        # RETURN ALL 56 FEATURES IN CORRECT ORDER
        # ═══════════════════════════════════════════════════════════════
        return [
            temp, prcp, wspd, is_holiday, day_of_week, is_weekend,
            days_to_holiday, days_since_holiday, hour, day, month, year, quarter,
            hour_sin, hour_cos, month_sin, month_cos, day_sin, day_cos,
            is_morning_rush, is_evening_rush, is_rush_hour, is_business_hours,
            time_of_day, season, week_of_year, is_month_start, is_month_end,
            days_since_epoch,
            lag_1h, lag_2h, lag_3h, lag_6h, lag_12h, lag_24h, lag_48h, lag_168h,
            roll_mean_4h, roll_std_4h, roll_min_4h, roll_max_4h,
            roll_mean_12h, roll_std_12h, roll_min_12h, roll_max_12h,
            roll_mean_24h, roll_std_24h, roll_min_24h, roll_max_24h,
            roll_mean_168h, roll_std_168h, roll_min_168h, roll_max_168h,
            ema_24h, demand_change_1h, demand_change_24h
        ]

# Global instance
prediction_service = None

def get_prediction_service():
    global prediction_service
    if prediction_service is None:
        prediction_service = PredictionService()
    return prediction_service
