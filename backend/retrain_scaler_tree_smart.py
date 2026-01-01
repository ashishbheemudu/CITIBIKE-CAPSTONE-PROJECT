
import logging
import sys
import os
import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import random

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from prediction_service import PredictionService, BASE_DIR, MODELS_DIR

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("SmartScalerRetrainer")

class MockScaler:
    def __init__(self):
        self.captured_features = []
        
    def transform(self, X):
        # Capture the features!
        # X is (1, n_features)
        self.captured_features.extend(X)
        # Return dummy data to let the code proceed (it will crash in model.predict, but we catch it)
        return X

def retrain_smart():
    logger.info("🚀 Starting SMART Scaler Retraining...")
    
    # 1. Instantiate Service
    ps = PredictionService()
    
    # 2. Force Load Data (The 11M rows)
    logger.info("📊 Loading Data...")
    ps._lazy_load_data()
    
    # 3. Force Load Models (so we can overwrite the scaler AFTER)
    ps._lazy_load_models()
    
    if ps.historical_data is None or ps.historical_data.empty:
        logger.error("❌ Failed to load historical data!")
        return

    # 3. Inject Mock Scaler
    mock_scaler = MockScaler()
    ps.scalers['tree'] = mock_scaler
    logger.info("🕵️ Mock Scaler Injected. Harvesting features...")

    # 4. Generate Features for Random Stations/Times
    stations = ps.historical_data['station_name'].unique()
    sampled_stations = random.sample(list(stations), min(20, len(stations)))
    
    # Generate for different times of year to catch seasonality
    dates = [
        "2023-01-15T12:00:00Z", "2023-04-10T08:00:00Z", 
        "2023-07-20T18:00:00Z", "2023-10-05T15:00:00Z"
    ]
    
    count = 0
    for station in sampled_stations:
        for date_str in dates:
            try:
                # We expect this to fail during model prediction step, 
                # but AFTER feature scaling step (which we intercepted)
                ps.predict(station, date_str, hours_ahead=24)
            except Exception:
                # Expected crash because model.predict gets unscaled data or we caused it
                pass
            
            count += 1
            if count % 10 == 0:
                logger.info(f"Harvested {len(mock_scaler.captured_features)} feature vectors...")

    # 5. Fit Real Scaler
    if not mock_scaler.captured_features:
        logger.error("❌ No features captured!")
        return
        
    logger.info(f"✅ Harvested {len(mock_scaler.captured_features)} total feature vectors.")
    
    X = np.array(mock_scaler.captured_features)
    
    logger.info("🔄 Retraining Feature Scaler (tree)...")
    real_scaler = StandardScaler()
    real_scaler.fit(X)
    
    save_path = os.path.join(MODELS_DIR, "scaler_tree.save")
    joblib.dump(real_scaler, save_path)
    logger.info(f"✅ Saved NEW scaler_tree.save (Mean[0]: {real_scaler.mean_[0]:.4f})")
    
    logger.info("🎉 Smart Retraining Complete! Reboot server.")

if __name__ == "__main__":
    retrain_smart()
