
import pandas as pd
import joblib
import os
from sklearn.preprocessing import StandardScaler
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ScalerRetrainer")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "v1_core", "final_station_demand_robust_features.parquet")
MODELS_DIR = os.path.join(BASE_DIR, "models")

def retrain_scalers():
    logger.info("🚀 Starting Scaler Retraining on Server...")
    
    if not os.path.exists(DATA_PATH):
        logger.error(f"❌ Data file not found: {DATA_PATH}")
        return

    # Load Data
    logger.info("📊 Loading 11M rows of historical data...")
    # Only load columns needed for scaling to save RAM
    df = pd.read_parquet(DATA_PATH, columns=['pickups', 'temp', 'prcp', 'wspd', 'station_name'])
    
    logger.info(f"✅ Data Loaded: {len(df)} rows")

    # 1. Retrain Target Scaler (pickups)
    logger.info("🔄 Retraining Target Scaler (y)...")
    scaler_y = StandardScaler()
    scaler_y.fit(df[['pickups']])
    
    joblib.dump(scaler_y, os.path.join(MODELS_DIR, "scaler_y.save"))
    logger.info(f"✅ Saved scaler_y.save (Mean: {scaler_y.mean_[0]:.4f}, Scale: {scaler_y.scale_[0]:.4f})")

    # 2. Retrain Feature Scaler (tree)
    # Features used: 'station_name' (encoded?), 'temp', 'prcp', 'wspd', 'hour', 'month', 'day_of_week'
    # Wait, the feature scaler expects a specific set of columns. 
    # If we don't know the exact columns the model expects, we shouldn't touch scaler_tree unless we are sure.
    # The current issue is specifically "Scaler output too low", which is scaler_y.
    # So we focus on scaler_y.
    
    logger.info("🎉 Scaler Retraining Complete!")

if __name__ == "__main__":
    retrain_scalers()
