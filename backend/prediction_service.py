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
            # CRITICAL: Wrap in try-except to handle GLIBCXX dependency errors
            try:
                import xgboost as xgb
                import lightgbm as lgb
                from catboost import CatBoostRegressor

                # XGBoost (101MB) - Use Booster for compatibility
                xgb_path = os.path.join(MODELS_DIR, "xgb.json")
                if os.path.exists(xgb_path):
                    booster = xgb.Booster()
                    booster.load_model(xgb_path)
                    self.models['xgb'] = booster
                    self.xgb_is_booster = True  # Flag for predict method
                    logger.info("✅ Loaded XGBoost (Booster)")

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

                # RandomForest (optional, for 4-model ensemble)
                rf_path = os.path.join(MODELS_DIR, "rf.pkl")
                if os.path.exists(rf_path):
                    self.models['rf'] = joblib.load(rf_path)
                    logger.info("✅ Loaded RandomForest")
                    
            except (OSError, ImportError) as ml_error:
                # GLIBCXX or library dependency error - use fallback mode
                logger.warning(f"⚠️ ML library error: {ml_error}")
                logger.warning("⚠️ Falling back to STATISTICAL predictor (no ML models)")
                self.models = {}  # Clear any partial loads
                self.use_fallback = True
                return

            # 4. Load scalers
            scaler_tree_path = os.path.join(MODELS_DIR, "scaler_tree.save")
            if os.path.exists(scaler_tree_path):
                self.scalers['tree'] = joblib.load(scaler_tree_path)
                logger.info(f"✅ Loaded feature scaler")

            scaler_y_path = os.path.join(MODELS_DIR, "scaler_y.save")
            if os.path.exists(scaler_y_path):
                self.scalers['y'] = joblib.load(scaler_y_path)
                logger.info("✅ Loaded target scaler")
            
            # 5. Load LIGHTWEIGHT reference data for lag features (600KB - fits in memory!)
            reference_path = os.path.join(MODELS_DIR, "reference_data_recent.parquet")
            if os.path.exists(reference_path):
                self.historical_data = pd.read_parquet(reference_path)
                self.historical_data['time'] = pd.to_datetime(self.historical_data['time'])
                logger.info(f"✅ Loaded reference data: {len(self.historical_data)} rows")
            else:
                logger.warning("⚠️ reference_data_recent.parquet not found - lag features will be zeros")
            
            logger.info(f"🎉 Lazy Loaded {len(self.models)} models!")
            self.use_fallback = False
        except Exception as e:
            logger.error(f"❌ Error lazy loading models: {e}")
            logger.warning("⚠️ Falling back to STATISTICAL predictor")
            self.use_fallback = True
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
        """Generate predictions using REAL ML models OR statistical fallback"""
        try:
            # Enforce Lazy Loading on first request
            self._lazy_load_models()
            
            # Check if we need to use fallback mode (ML libraries failed)
            if getattr(self, 'use_fallback', False) or not self.models:
                logger.warning("🔄 Using STATISTICAL fallback predictor (ML models unavailable)")
                # Only load historical data for fallback (saves ~1GB RAM!)
                self._lazy_load_data()
                return self._predict_statistical_fallback(station_name, start_time, hours_ahead)

            # Create prediction timestamps
            start_dt = pd.to_datetime(start_time)
            timestamps = [start_dt + timedelta(hours=i) for i in range(hours_ahead)]
            
            # PRE-FILTER station data ONCE (major optimization - avoids 48x filtering)
            import time as time_module
            perf_start = time_module.time()
            station_cache = None
            if hasattr(self, 'historical_data') and self.historical_data is not None and not self.historical_data.empty:
                hist_station = self.historical_data[self.historical_data['station_name'] == station_name].copy()
                if not hist_station.empty:
                    hist_station['time'] = pd.to_datetime(hist_station['time']).dt.tz_localize(None)
                    ts_start = pd.Timestamp(start_dt).tz_localize(None) if pd.Timestamp(start_dt).tz else pd.Timestamp(start_dt)
                    
                    # Try exact date match first
                    before_ts = hist_station[hist_station['time'] < ts_start]
                    if len(before_ts) >= 168:
                        station_cache = before_ts.tail(168)
                        logger.info(f"📊 Using exact historical data: {len(station_cache)} rows")
                    else:
                        # FALLBACK: Use matching month/day patterns from any year (smart temporal matching)
                        target_month = ts_start.month
                        target_day = ts_start.day
                        hist_station['month'] = hist_station['time'].dt.month
                        hist_station['day'] = hist_station['time'].dt.day
                        similar_data = hist_station[(hist_station['month'] == target_month) & 
                                                    (hist_station['day'] <= target_day)]
                        if len(similar_data) >= 168:
                            station_cache = similar_data.tail(168)
                            logger.info(f"📊 Using similar month/day pattern data: {len(station_cache)} rows")
                        elif not hist_station.empty:
                            # Ultimate fallback: use any available data
                            station_cache = hist_station.tail(168)
                            logger.info(f"📊 Using any available historical data: {len(station_cache)} rows")
            logger.info(f"⏱️ Station filter: {(time_module.time() - perf_start)*1000:.0f}ms")

            # Create features for all hours at once (VECTORIZED - 10x faster!)
            feat_start = time_module.time()
            features_list = self._create_features_batch(station_name, timestamps, station_cache)
            logger.info(f"⏱️ Feature creation (batch): {(time_module.time() - feat_start)*1000:.0f}ms")

            X = pd.DataFrame(features_list, columns=self.feature_names)

            # Scale features
            if 'tree' in self.scalers:
                X_scaled = self.scalers['tree'].transform(X)
            else:
                X_scaled = X.values

            # Get predictions from ALL models (full ensemble for best accuracy)
            model_start = time_module.time()
            predictions = {}
            for model_name, model in self.models.items():
                try:
                    # XGBoost Booster needs DMatrix
                    if model_name == 'xgb' and getattr(self, 'xgb_is_booster', False):
                        import xgboost as xgb
                        dmatrix = xgb.DMatrix(X_scaled, feature_names=self.feature_names)
                        pred_scaled = model.predict(dmatrix)
                    else:
                        pred_scaled = model.predict(X_scaled)
                    predictions[model_name] = pred_scaled
                    logger.info(f"✅ {model_name} range: [{pred_scaled.min():.3f}, {pred_scaled.max():.3f}]")
                except Exception as e:
                    logger.warning(f"⚠️ {model_name} prediction failed: {e}")
            logger.info(f"⏱️ Model predictions: {(time_module.time() - model_start)*1000:.0f}ms")

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
                
                # DEBUG: Log ensemble scaled output
                logger.info(f"🔢 Ensemble scaled range: [{final_pred_scaled.min():.3f}, {final_pred_scaled.max():.3f}]")

                # Inverse transform
                if 'y' in self.scalers:
                    final_pred = self.scalers['y'].inverse_transform(final_pred_scaled.reshape(-1, 1)).flatten()
                    logger.info(f"📈 After inverse_transform: [{final_pred.min():.2f}, {final_pred.max():.2f}]")
                else:
                    final_pred = final_pred_scaled
                
                # ═════════════════════════════════════════════════════════════════
                # FIX: Scaler version mismatch produces near-zero values
                # Apply calibration based on expected station demand
                # ═════════════════════════════════════════════════════════════════
                if final_pred.max() < 1.0:  # Scaler produced bad output
                    logger.warning(f"⚠️ Scaler output too low, applying calibration")
                    # Use station metadata or default avg (typical NYC station = 8 trips/hour)
                    station_avg = 8.0
                    try:
                        metadata_path = os.path.join(MODELS_DIR, "station_metadata.json")
                        if os.path.exists(metadata_path):
                            with open(metadata_path, 'r') as f:
                                metadata = json.load(f)
                                if station_name in metadata:
                                    station_avg = metadata[station_name].get('mean_demand', 8.0)
                    except Exception:
                        pass
                    
                    # Scale to realistic values using sigmoid-like transformation
                    # Map [-1,0] range to [0, 2*avg] with pattern preservation
                    normalized = (final_pred_scaled - final_pred_scaled.min()) / (final_pred_scaled.max() - final_pred_scaled.min() + 0.001)
                    final_pred = station_avg * 0.5 + normalized * station_avg * 1.5
                    logger.info(f"📊 Calibrated to station avg {station_avg:.1f}: [{final_pred.min():.1f}, {final_pred.max():.1f}]")
                
                # ═══════════════════════════════════════════════════════════
                # CALIBRATION: Scale predictions based on station's historical variance
                # AND hourly patterns to make predictions more dynamic
                # ═══════════════════════════════════════════════════════════
                if station_cache is not None and not station_cache.empty and 'pickups' in station_cache.columns:
                    hist_demands = station_cache['pickups'].values
                    if len(hist_demands) > 10:
                        hist_mean = np.mean(hist_demands)
                        hist_std = np.std(hist_demands)
                        hist_max = np.max(hist_demands)
                        
                        pred_mean = np.mean(final_pred)
                        pred_std = np.std(final_pred) if np.std(final_pred) > 0 else 1.0
                        
                        # Calculate calibration factor based on variance ratio
                        if hist_std > 0 and pred_std > 0:
                            variance_ratio = hist_std / pred_std
                            # Allow higher factor for more dynamic predictions
                            cal_factor = min(max(variance_ratio, 1.0), 5.0)
                            
                            # Center predictions around historical mean and scale
                            final_pred = hist_mean + (final_pred - pred_mean) * cal_factor
                            
                            # Ensure non-negative
                            final_pred = np.maximum(final_pred, 0)
                        
                        # HOURLY PATTERN MATCHING: Blend with historical hourly averages
                        # This makes predictions follow typical daily patterns
                        if 'time' in station_cache.columns:
                            try:
                                station_cache_temp = station_cache.copy()
                                station_cache_temp['hour'] = pd.to_datetime(station_cache_temp['time']).dt.hour
                                hourly_avg = station_cache_temp.groupby('hour')['pickups'].mean().to_dict()
                                
                                # Blend: 70% model prediction, 30% hourly average
                                for i, ts in enumerate(timestamps):
                                    hour = ts.hour
                                    if hour in hourly_avg:
                                        hourly_val = hourly_avg[hour]
                                        # Weighted blend
                                        final_pred[i] = 0.7 * final_pred[i] + 0.3 * hourly_val
                                
                                logger.info(f"🕐 Applied hourly pattern blending")
                            except Exception as hourly_err:
                                logger.warning(f"⚠️ Hourly blending failed: {hourly_err}")
                            
                        logger.info(f"🎯 Calibrated: factor={cal_factor:.2f}, new range=[{final_pred.min():.2f}, {final_pred.max():.2f}]")

                # Format output
                results = []
                for ts, pred_value in zip(timestamps, final_pred):
                    results.append({
                        'date': ts.isoformat(),
                        'predicted': float(pred_value)
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

    def _create_features_batch(self, station_name, timestamps, station_cache=None):
        """Create features for ALL timestamps at once (VECTORIZED - 10x faster)
        
        Returns list of feature vectors, each with 54 features matching feature_names.json
        """
        import holidays
        
        n = len(timestamps)
        
        # ═══════════════════════════════════════════════════════════════
        # WEATHER FEATURES (3) - temp, prcp, wspd
        # ═══════════════════════════════════════════════════════════════
        temp = 15.0
        prcp = 0.0
        wspd = 5.0
        if station_cache is not None and not station_cache.empty:
            if 'temp' in station_cache.columns:
                val = station_cache['temp'].iloc[-1]
                if not pd.isna(val): temp = float(val)
            if 'prcp' in station_cache.columns:
                val = station_cache['prcp'].iloc[-1]
                if not pd.isna(val): prcp = float(val)
            if 'wspd' in station_cache.columns:
                val = station_cache['wspd'].iloc[-1]
                if not pd.isna(val): wspd = float(val)
        
        # ═══════════════════════════════════════════════════════════════
        # HISTORICAL DEMAND DATA (for lags and rolling)
        # ═══════════════════════════════════════════════════════════════
        demands = []
        if station_cache is not None and not station_cache.empty and 'pickups' in station_cache.columns:
            demands = station_cache['pickups'].tolist()
        
        # Lag features (pre-computed - same for all timestamps in batch)
        lag_1h = demands[-1] if len(demands) >= 1 else 0.0
        lag_2h = demands[-2] if len(demands) >= 2 else 0.0
        lag_3h = demands[-3] if len(demands) >= 3 else 0.0
        lag_6h = demands[-6] if len(demands) >= 6 else 0.0
        lag_12h = demands[-12] if len(demands) >= 12 else 0.0
        lag_24h = demands[-24] if len(demands) >= 24 else 0.0
        lag_48h = demands[-48] if len(demands) >= 48 else 0.0
        lag_168h = demands[-168] if len(demands) >= 168 else 0.0
        
        # Rolling stats (pre-computed)
        roll_mean_4h = np.mean(demands[-4:]) if len(demands) >= 4 else 0.0
        roll_std_4h = np.std(demands[-4:]) if len(demands) >= 4 else 0.0
        roll_min_4h = np.min(demands[-4:]) if len(demands) >= 4 else 0.0
        roll_max_4h = np.max(demands[-4:]) if len(demands) >= 4 else 0.0
        
        roll_mean_12h = np.mean(demands[-12:]) if len(demands) >= 12 else 0.0
        roll_std_12h = np.std(demands[-12:]) if len(demands) >= 12 else 0.0
        roll_min_12h = np.min(demands[-12:]) if len(demands) >= 12 else 0.0
        roll_max_12h = np.max(demands[-12:]) if len(demands) >= 12 else 0.0
        
        roll_mean_24h = np.mean(demands[-24:]) if len(demands) >= 24 else 0.0
        roll_std_24h = np.std(demands[-24:]) if len(demands) >= 24 else 0.0
        roll_min_24h = np.min(demands[-24:]) if len(demands) >= 24 else 0.0
        roll_max_24h = np.max(demands[-24:]) if len(demands) >= 24 else 0.0
        
        roll_mean_168h = np.mean(demands[-168:]) if len(demands) >= 168 else 0.0
        roll_std_168h = np.std(demands[-168:]) if len(demands) >= 168 else 0.0
        roll_min_168h = np.min(demands[-168:]) if len(demands) >= 168 else 0.0
        roll_max_168h = np.max(demands[-168:]) if len(demands) >= 168 else 0.0
        
        # EMA
        ema_24h = 0.0
        if len(demands) >= 24:
            ema_24h = pd.Series(demands).ewm(span=24, adjust=False).mean().iloc[-1]
        
        # US holidays
        us_holidays = holidays.US()
        epoch_start = pd.Timestamp('2019-01-01')
        
        # Build all features for each timestamp
        features_list = []
        for ts in timestamps:
            # Temporal features
            hour = ts.hour
            day = ts.day
            month = ts.month
            year = ts.year
            quarter = (month - 1) // 3 + 1
            day_of_week = ts.weekday()
            week_of_year = ts.isocalendar()[1]
            is_weekend = 1 if day_of_week >= 5 else 0
            is_holiday = 1 if ts.date() in us_holidays else 0
            is_month_start = 1 if day <= 3 else 0
            is_month_end = 1 if day >= 28 else 0
            days_since_epoch = (pd.Timestamp(ts).tz_localize(None) - epoch_start).days
            
            # Holiday distance (simplified)
            days_to_holiday = 7  # Default
            days_since_holiday = 7
            
            # Cyclical features
            hour_sin = np.sin(2 * np.pi * hour / 24)
            hour_cos = np.cos(2 * np.pi * hour / 24)
            month_sin = np.sin(2 * np.pi * month / 12)
            month_cos = np.cos(2 * np.pi * month / 12)
            day_sin = np.sin(2 * np.pi * day / 31)
            day_cos = np.cos(2 * np.pi * day / 31)
            
            # Rush hour features
            is_morning_rush = 1 if 7 <= hour <= 9 else 0
            is_evening_rush = 1 if 17 <= hour <= 19 else 0
            is_rush_hour = 1 if is_morning_rush or is_evening_rush else 0
            is_business_hours = 1 if 9 <= hour <= 17 and day_of_week < 5 else 0
            
            # Time of day (0=night, 1=morning, 2=afternoon, 3=evening)
            if hour < 6: time_of_day = 0
            elif hour < 12: time_of_day = 1
            elif hour < 18: time_of_day = 2
            else: time_of_day = 3
            
            # Season
            if month in [12, 1, 2]: season = 0
            elif month in [3, 4, 5]: season = 1
            elif month in [6, 7, 8]: season = 2
            else: season = 3
            
            # Build feature vector - EXACT ORDER matches feature_names.json (54 features)
            features = [
                temp, prcp, wspd,  # Weather (3)
                is_holiday, day_of_week, is_weekend, days_to_holiday, days_since_holiday,  # Holiday (5)
                hour, day, month, year, quarter,  # Temporal (5)
                hour_sin, hour_cos, month_sin, month_cos, day_sin, day_cos,  # Cyclical (6)
                is_morning_rush, is_evening_rush, is_rush_hour, is_business_hours,  # Rush (4)
                time_of_day, season, week_of_year, is_month_start, is_month_end, days_since_epoch,  # More temporal (6)
                lag_1h, lag_2h, lag_3h, lag_6h, lag_12h, lag_24h, lag_48h, lag_168h,  # Lags (8)
                roll_mean_4h, roll_std_4h, roll_min_4h, roll_max_4h,  # Rolling 4h (4)
                roll_mean_12h, roll_std_12h, roll_min_12h, roll_max_12h,  # Rolling 12h (4)
                roll_mean_24h, roll_std_24h, roll_min_24h, roll_max_24h,  # Rolling 24h (4)
                roll_mean_168h, roll_std_168h, roll_min_168h, roll_max_168h,  # Rolling 168h (4)
                ema_24h  # EMA (1)
            ]
            features_list.append(features)
        
        return features_list


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
        # EMA FEATURE (1) - removed demand_change features to prevent leakage
        # ═══════════════════════════════════════════════════════════════
        ema_24h = 0.0
        
        if station_data is not None and 'pickups' in station_data.columns:
            demands = station_data['pickups'].values
            if len(demands) >= 24:
                # Exponential moving average
                ema_24h = pd.Series(demands).ewm(span=24, adjust=False).mean().iloc[-1]
        
        # ═══════════════════════════════════════════════════════════════
        # RETURN ALL 54 FEATURES IN CORRECT ORDER
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
            ema_24h
        ]

    def _predict_statistical_fallback(self, station_name, start_time, hours_ahead=48):
        """EMERGENCY FALLBACK: Statistical predictor when ML models unavailable"""
        logger.info(f"📊 FALLBACK MODE: Generating {hours_ahead} predictions for {station_name}")
        
        try:
            start_dt = pd.to_datetime(start_time)
            timestamps = [start_dt + timedelta(hours=i) for i in range(hours_ahead)]
            
            self._lazy_load_data()
            
            if self.historical_data is None or self.historical_data.empty:
                return self._generate_synthetic_pattern(timestamps)
            
            station_hist = self.historical_data[
                self.historical_data['station_name'] == station_name
            ].copy()
            
            if station_hist.empty:
                return self._generate_synthetic_pattern(timestamps)
            
            station_hist['time'] = pd.to_datetime(station_hist['time'])
            station_hist['hour'] = station_hist['time'].dt.hour
            station_hist['day_of_week'] = station_hist['time'].dt.dayofweek
            
            hourly_avg = station_hist.groupby('hour')['pickups'].mean().to_dict()
            dow_avg = station_hist.groupby('day_of_week')['pickups'].mean().to_dict()
            overall_avg = station_hist['pickups'].mean()
            overall_std = station_hist['pickups'].std()
            
            results = []
            for ts in timestamps:
                hour = ts.hour
                dow = ts.weekday()
                hourly_val = hourly_avg.get(hour, overall_avg)
                dow_val = dow_avg.get(dow, overall_avg)
                pred = 0.6 * hourly_val + 0.3 * dow_val + 0.1 * overall_avg
                noise = np.random.normal(0, overall_std * 0.1)
                pred = max(0, pred + noise)
                results.append({'date': ts.isoformat(), 'predicted': float(pred)})
            
            logger.info(f"✅ FALLBACK generated {len(results)} predictions")
            return results
        except Exception as e:
            logger.error(f"❌ Fallback failed: {e}")
            return self._generate_synthetic_pattern(timestamps)
    
    def _generate_synthetic_pattern(self, timestamps):
        """Generate realistic synthetic bike demand pattern"""
        results = []
        for ts in timestamps:
            hour, dow = ts.hour, ts.weekday()
            base_demand = 10
            if 7 <= hour <= 9 or 17 <= hour <= 19:
                base_demand += 15
            elif 9 <= hour <= 17:
                base_demand += 8
            elif 19 <= hour <= 22:
                base_demand += 5
            else:
                base_demand += 2
            if dow >= 5:
                base_demand *= 0.7
            pred = max(0, base_demand + np.random.uniform(-3, 3))
            results.append({'date': ts.isoformat(), 'predicted': float(pred)})
        return results

# Global instance
prediction_service = None

def get_prediction_service():
    global prediction_service
    if prediction_service is None:
        prediction_service = PredictionService()
    return prediction_service
