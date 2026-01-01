
import pandas as pd
import joblib
import os
import json
from sklearn.preprocessing import StandardScaler
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ScalerRetrainer")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "v1_core", "final_station_demand_robust_features.parquet")
MODELS_DIR = os.path.join(BASE_DIR, "models")
FEATURE_NAMES_PATH = os.path.join(MODELS_DIR, "feature_names.json")

def retrain_scalers():
    logger.info("🚀 Starting Advanced Scaler Retraining on Server...")
    
    if not os.path.exists(DATA_PATH):
        logger.error(f"❌ Data file not found: {DATA_PATH}")
        return
        
    if not os.path.exists(FEATURE_NAMES_PATH):
        logger.error(f"❌ Feature names file not found: {FEATURE_NAMES_PATH}")
        return

    # Load Feature Names
    with open(FEATURE_NAMES_PATH, 'r') as f:
        feature_names = json.load(f)
    logger.info(f"📋 Found {len(feature_names)} features in model definition.")

    # Load Data
    logger.info("📊 Loading 11M rows of historical data...")
    # Load ALL columns to ensure we have all features
    df = pd.read_parquet(DATA_PATH)
    logger.info(f"✅ Data Loaded: {len(df)} rows")

    # 1. Retrain Target Scaler (pickups)
    logger.info("🔄 Retraining Target Scaler (y)...")
    scaler_y = StandardScaler()
    scaler_y.fit(df[['pickups']])
    joblib.dump(scaler_y, os.path.join(MODELS_DIR, "scaler_y.save"))
    logger.info(f"✅ Saved scaler_y.save (Mean: {scaler_y.mean_[0]:.4f}, Scale: {scaler_y.scale_[0]:.4f})")

    # 2. Retrain Feature Scaler (tree)
    logger.info("🔄 Retraining Feature Scaler (tree)...")
    
    # Check if all features exist in dataframe
    missing_features = [f for f in feature_names if f not in df.columns]
    if missing_features:
        logger.warning(f"⚠️ Missing features in parquet: {missing_features}")
        logger.warning("Attempting to proceed with available features (this might cause issues)...")
    
    # Select features in the exact order
    X = df[feature_names]
    
    scaler_tree = StandardScaler()
    scaler_tree.fit(X)
    joblib.dump(scaler_tree, os.path.join(MODELS_DIR, "scaler_tree.save"))
    logger.info(f"✅ Saved scaler_tree.save")

    logger.info("🎉 Full Scaler Retraining Complete! Reboot the server now.")

if __name__ == "__main__":
    retrain_scalers()
