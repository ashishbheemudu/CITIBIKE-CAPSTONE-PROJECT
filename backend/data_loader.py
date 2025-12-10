import pandas as pd
import os
import joblib
import glob
import json

# Define base paths
# Legacy path for backward compatibility
LEGACY_DATA_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../csv files"))
# New structured path for God-Level assets
DATA_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "data"))

class DataLoader:
    def __init__(self):
        self.hourly_demand = None
        self.daily_demand = None
        self.top_stations = None
        self.station_hourly_demand = None
        self.top_routes = None
        self.anomalies = None
        self.station_locations = None
        self.model = None 
        self.robust_features = None
        self.scaler = None
        self.weather_cache = {}
        
        # Advanced Assets (Lazy Loaded)
        self.v8_abm_data = None
        self.v8_tft_forecasts = None
        self.v8_rl_policy = None
        self.v8_equity_scores = None
        self.v8_digital_twin = None
        
        self.v9_uhi_data = None
        self.v9_safety_events = None
        self.v9_synthetic_trips = None
        self.v9_bikecoin_data = None

    def load_core_data(self):
        print("Loading core data...")
        try:
            # Try loading from new structure first, fall back to legacy
            self._load_core_csvs()
            
            # Load Parquet
            self._load_parquet()
            
            # Load Scaler
            self._load_scaler()

            # Pre-fetch Weather
            self._fetch_weather()

            print("Core Data loaded successfully.")
        except Exception as e:
            print(f"Error loading data: {e}")

    def _load_core_csvs(self):
        # Helper to try multiple paths
        def read_csv_safe(filename, folder="v1_core"):
            # 1. Try new structure
            p1 = os.path.join(DATA_ROOT, folder, filename)
            if os.path.exists(p1): return pd.read_csv(p1)
            # 2. Try legacy
            p2 = os.path.join(LEGACY_DATA_PATH, filename)
            if os.path.exists(p2): return pd.read_csv(p2)
            return None

        print("Loading hourly_demand_timekey.csv...")
        self.hourly_demand = read_csv_safe("hourly_demand_timekey.csv")
        if self.hourly_demand is not None:
            self.hourly_demand['hour'] = pd.to_datetime(self.hourly_demand['hour'])
            
        print("Loading daily_demand.csv...")
        self.daily_demand = read_csv_safe("daily_demand.csv")
        
        print("Loading top_stations.csv...")
        self.top_stations = read_csv_safe("top_stations.csv")
        if self.top_stations is not None:
            self.top_stations.rename(columns={'start_station_name': 'station_name'}, inplace=True)
            
        print("Loading station_hourly_demand.csv...")
        self.station_hourly_demand = read_csv_safe("station_hourly_demand.csv")
        
        print("Loading top_200_routes_map.csv...")
        self.top_routes = read_csv_safe("top_200_routes_map.csv")
        
        print("Loading anomalies.csv...")
        self.anomalies = read_csv_safe("anomalies.csv")
        if self.anomalies is None: self.anomalies = pd.DataFrame()

        print("Loading all_stations_locations.csv...")
        self.station_locations = read_csv_safe("all_stations_locations.csv")
        if self.station_locations is not None:
            self.station_locations.rename(columns={
                'start_station_name': 'station_name',
                'start_lat': 'lat',
                'start_lng': 'lon'
            }, inplace=True)

    def _load_parquet(self):
        # HYDRA PATHS
        # 1. Hydra Deployment Path (backend/models/reference_data_recent.parquet)
        p_hydra = os.path.join(os.path.dirname(__file__), "models", "reference_data_recent.parquet")
        
        # 2. Legacy Data Paths
        p1 = os.path.join(DATA_ROOT, "v1_core", "final_station_demand_robust_features.parquet")
        p2 = os.path.join(LEGACY_DATA_PATH, "final_station_demand_robust_features.parquet")
        
        path = None
        if os.path.exists(p1):
            path = p1
            print(f"✅ Found Full Historical Data: {path}")
        elif os.path.exists(p_hydra):
            path = p_hydra
            print(f"⚠️ Full data missing, using Hydra Reference Data (Recent only): {path}")
        elif os.path.exists(p2):
            path = p2
        
        if path:
            print(f"Loading robust features from {path}...")
            self.robust_features = pd.read_parquet(path)
            self.robust_features['time'] = pd.to_datetime(self.robust_features['time'])
            
            # Populate Weather Cache from Robust Features (Avoids Meteostat calls)
            print("Populating weather cache from robust features...")
            weather_df = self.robust_features[['time', 'temp', 'prcp']].drop_duplicates(subset=['time'])
            weather_df['date_str'] = weather_df['time'].dt.strftime('%Y-%m-%d')
            
            # Group by date to get daily average/sum
            daily_weather = weather_df.groupby('date_str').agg({
                'temp': 'mean',
                'prcp': 'sum'
            }).reset_index()
            
            # Add Mock Humidity (Simple heuristic: higher if raining, slightly higher in summer)
            def estimate_humidity(row):
                base_humidity = 55 # Base level
                if row['prcp'] > 0:
                    base_humidity += 25 # Rain adds humidity
                # Add some random variation based on date hash to make it consistent
                date_hash = hash(row['date_str']) % 20 
                return min(100, max(30, base_humidity + date_hash))

            daily_weather['humidity'] = daily_weather.apply(estimate_humidity, axis=1)
            
            self.weather_cache = daily_weather.set_index('date_str').rename(columns={'temp': 'tavg'}).to_dict(orient='index')
            print(f"Weather cache populated with {len(self.weather_cache)} days.")
            
            # --- HYDRA AUTO-RECOVERY ---
            # If CSVs failed to load station_locations, extract it from Parquet!
            if self.station_locations is None or self.station_locations.empty:
                print("🛠️ Generating Station List from Hydra Data (CSV missing)...")
                # Hydra data has 'station_name' but possibly NOT lat/lon if they were dropped to save space?
                # Hydra 'reference_data_recent.parquet' usually keeps metadata.
                # If lat/lon missing, we just use names.
                # Check for lat/lon columns
                cols = self.robust_features.columns
                if 'start_lat' in cols and 'start_lng' in cols:
                     unique_stations = self.robust_features[['station_name', 'start_lat', 'start_lng']].drop_duplicates(subset=['station_name'])
                     unique_stations.rename(columns={'start_lat': 'lat', 'start_lng': 'lon'}, inplace=True)
                     self.station_locations = unique_stations
                else:
                     # Just names
                     unique_stations = pd.DataFrame(self.robust_features['station_name'].unique(), columns=['station_name'])
                     # Add dummy lat/lon to prevent deck.gl crash (NYC Center)
                     unique_stations['lat'] = 40.7128
                     unique_stations['lon'] = -74.0060
                     self.station_locations = unique_stations
                
                print(f"✅ Recovered {len(self.station_locations)} stations from Parquet.")
        else:
            print("Robust features parquet not found.")

    def _load_scaler(self):
        # Professional Path
        p1 = os.path.join(os.path.dirname(__file__), "models", "scaler_advanced.save")
        # Legacy Path (just in case)
        p2 = os.path.join(LEGACY_DATA_PATH, "scaler_advanced.save")
        
        path = p1 if os.path.exists(p1) else (p2 if os.path.exists(p2) else None)
        
        if path:
            self.scaler = joblib.load(path)
            print("Scaler loaded.")
        else:
            print("Scaler not found.")

    def _fetch_weather(self):
        if self.hourly_demand is None: return
        print("Fetching weather data from Meteostat...")
        try:
            from meteostat import Point, Daily
            nyc = Point(40.7128, -74.0060)
            start = self.hourly_demand['hour'].min()
            end = self.hourly_demand['hour'].max()
            weather = Daily(nyc, start, end).fetch()
            weather['date_str'] = weather.index.strftime('%Y-%m-%d')
            
            # Helper to estimate humidity (Meteostat free tier might miss it)
            def estimate_humidity(row):
                base_humidity = 55 # Base level
                if row['prcp'] > 0:
                    base_humidity += 25 # Rain adds humidity
                # Add some random variation based on date hash to make it consistent
                date_hash = hash(row['date_str']) % 20 
                return min(100, max(30, base_humidity + date_hash))

            if 'tavg' in weather.columns and 'prcp' in weather.columns:
                weather['humidity'] = weather.apply(estimate_humidity, axis=1)
                self.weather_cache = weather[['date_str', 'tavg', 'prcp', 'humidity']].set_index('date_str').to_dict(orient='index')
                print(f"✅ Weather cache populated with {len(self.weather_cache)} days via Meteostat.")
            else:
                print("⚠️ Meteostat returned incomplete data.")
                
        except Exception as e:
            print(f"Weather fetch error: {e}")

    def load_fallback_model(self):
        p1 = os.path.join(DATA_ROOT, "v1_core", "fallback_model.joblib")
        p2 = os.path.join(LEGACY_DATA_PATH, "fallback_model.joblib")
        path = p1 if os.path.exists(p1) else (p2 if os.path.exists(p2) else None)
        
        if path:
            self.fallback_model = joblib.load(path)
            print("Fallback model loaded.")
        else:
            self.fallback_model = None

    # --- GOD MODE LOADERS (V8) ---
    def load_v8_assets(self):
        print("Loading V8 God Protocol assets...")
        # Path used by generate_emergency_data.py
        v8_path = os.path.join(DATA_ROOT, "v8_abm")
        
        # ABM
        abm_file = os.path.join(v8_path, "abm_simulation.json")
        if os.path.exists(abm_file):
            with open(abm_file, 'r') as f: self.v8_abm_data = json.load(f)
            
        # TFT (Placeholder)
        tft_file = os.path.join(v8_path, "tft_forecasts.json")
        if os.path.exists(tft_file):
            with open(tft_file, 'r') as f: self.v8_tft_forecasts = json.load(f)
            
        # RL (Placeholder)
        rl_file = os.path.join(v8_path, "rl_policy.json")
        if os.path.exists(rl_file):
            with open(rl_file, 'r') as f: self.v8_rl_policy = json.load(f)
            
        # Equity (Placeholder)
        eq_file = os.path.join(v8_path, "equity_scores.json")
        if os.path.exists(eq_file):
            with open(eq_file, 'r') as f: self.v8_equity_scores = json.load(f)
            
        # Digital Twin (Placeholder)
        dt_file = os.path.join(v8_path, "digital_twin.json")
        if os.path.exists(dt_file):
            with open(dt_file, 'r') as f: self.v8_digital_twin = json.load(f)

    # --- OMNISCIENT MODE LOADERS (V9) ---
    def load_v9_assets(self):
        print("Loading V9 Omniscient Protocol assets...")
        # Path used by generate_emergency_data.py
        v9_path = os.path.join(DATA_ROOT, "v9_omniscient")
        
        # UHI
        uhi_file = os.path.join(v9_path, "uhi_analysis.json")
        if os.path.exists(uhi_file):
            with open(uhi_file, 'r') as f: self.v9_uhi_data = json.load(f)
            
        # Safety
        safe_file = os.path.join(v9_path, "safety_hotspots.json")
        if os.path.exists(safe_file):
            with open(safe_file, 'r') as f: self.v9_safety_events = json.load(f)
            
        # GAN (Placeholder)
        gan_file = os.path.join(v9_path, "synthetic_trips.json")
        if os.path.exists(gan_file):
            with open(gan_file, 'r') as f: self.v9_synthetic_trips = json.load(f)
            
        # Web3
        web3_file = os.path.join(v9_path, "web3_incentives.json")
        if os.path.exists(web3_file):
            with open(web3_file, 'r') as f: self.v9_bikecoin_data = json.load(f)

data_loader = DataLoader()
# data_loader.load_core_data() # Moved to main.py startup_event
# data_loader.load_fallback_model()
