
import os
import sys
import logging

# Add backend to path so imports work
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("SmokeTest")

def run_smoke_test():
    logger.info("🔥 Starting Smoke Test for Deployment...")

    try:
        # 1. Import DataLoader
        logger.info("Step 1: Importing DataLoader...")
        from data_loader import DataLoader
        
        # 2. Initialize DataLoader
        logger.info("Step 2: Initializing DataLoader...")
        dl = DataLoader()
        
        # 3. Load Core Data
        logger.info("Step 3: Loading Core Data (this might take a few seconds)...")
        # specific method to load just enough for health check
        dl.load_core_data()
        
        # 4. Assertions
        logger.info("Step 4: Verifying Data Integrity...")
        
        if dl.hourly_demand is None or dl.hourly_demand.empty:
            raise ValueError("❌ hourly_demand is Empty!")
        logger.info(f"✅ Hourly Demand Loaded: {len(dl.hourly_demand)} rows")
        
        if not dl.weather_cache:
            # It's okay if weather cache is lazy loaded, but usually it loads with core data
            logger.warning("⚠️ Weather cache is empty (might be lazy loaded)")
        else:
            logger.info(f"✅ Weather Cache Loaded: {len(dl.weather_cache)} days")

        # 5. Test Prediction Service Loading (Optional but good)
        logger.info("Step 5: outputting models directory file list for verification...")
        models_dir = os.path.join(os.path.dirname(__file__), "models")
        if os.path.exists(models_dir):
            logger.info(f"Models dir contents: {os.listdir(models_dir)}")
        else:
            logger.warning("⚠️ Models directory not found!")

        logger.info("🎉 SMOKE TEST PASSED! App is ready for deployment.")
        return True

    except Exception as e:
        logger.error(f"❌ SMOKE TEST FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = run_smoke_test()
    if not success:
        sys.exit(1)
