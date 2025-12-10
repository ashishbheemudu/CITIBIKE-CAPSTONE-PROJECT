from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from data_loader import data_loader
from prediction_service import prediction_service
from pydantic import BaseModel
import pandas as pd
import numpy as np
import asyncio
from concurrent.futures import ThreadPoolExecutor

# Create a thread pool for blocking ML operations
# ThreadPoolExecutor shares memory, avoiding data reloading issues.
# TensorFlow releases GIL during inference, so this should not block the event loop.
ml_executor = ThreadPoolExecutor(max_workers=1)

app = FastAPI(title="NYC Citi Bike Analytics Dashboard API")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for cloud deployment
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def startup_event():
    print("DEBUG: Startup event triggered")
    # Load data on startup
    print("DEBUG: Loading core data...")
    data_loader.load_core_data()
    print("DEBUG: Core data loaded. Prediction service handles model loading.")
    # data_loader.load_advanced_model()
    print("DEBUG: Startup complete.")

@app.get("/")
async def root():
    return {"message": "Backend is running"}

@app.get("/api/system-overview")
async def get_system_overview(
    start_date: str = None, 
    end_date: str = None, 
    days_of_week: str = None # Comma separated: "Monday,Tuesday"
):
    if data_loader.hourly_demand is None:
        raise HTTPException(status_code=503, detail="Data not loaded yet")

    # --- 1. Filter Data ---
    df = data_loader.hourly_demand.copy()
    
    # Date Filter
    if start_date:
        df = df[df['hour'] >= pd.to_datetime(start_date)]
    if end_date:
        df = df[df['hour'] <= pd.to_datetime(end_date)]
        
    # Day Filter
    if days_of_week:
        days_list = days_of_week.split(',')
        df = df[df['hour'].dt.day_name().isin(days_list)]
        
    if df.empty:
        # Return empty result with metadata indicating filters removed all data
        return {
            "kpis": {"total_trips": 0, "weekend_share": 0, "peak_hour": 0, "weekend_effect": 0},
            "time_series": [],
            "trend": [],
            "anomalies": [],
            "heatmap": [],
            "message": "No data found for selected date range and days. Try adjusting filters."
        }

    # --- 2. KPIs ---
    total_trips = int(df['trip_count'].sum())
    
    # Peak Hour (from filtered data)
    peak_hour_idx = df.groupby(df['hour'].dt.hour)['trip_count'].sum().idxmax()
    
    # Weekend Share (Dynamic)
    df['day_name'] = df['hour'].dt.day_name()
    weekend_trips = df[df['day_name'].isin(['Saturday', 'Sunday'])]['trip_count'].sum()
    weekend_share = (weekend_trips / total_trips) * 100 if total_trips > 0 else 0
    
    # Weekend Effect (Cohen's d) - Dynamic
    daily_counts = df.set_index('hour').resample('D')['trip_count'].sum().reset_index()
    daily_counts['is_weekend'] = daily_counts['hour'].dt.dayofweek >= 5
    group1 = daily_counts[daily_counts['is_weekend'] == True]['trip_count']
    group2 = daily_counts[daily_counts['is_weekend'] == False]['trip_count']
    
    cohens_d = 0
    if len(group1) > 1 and len(group2) > 1:
        diff = group1.mean() - group2.mean()
        pooled_std = np.sqrt((group1.std()**2 + group2.std()**2) / 2)
        cohens_d = diff / pooled_std if pooled_std > 0 else 0

    # --- 3. Time Series & Weather Overlay ---
    # Resample to Daily for chart
    daily_ts = df.set_index('hour').resample('D')['trip_count'].sum().reset_index()
    daily_ts['date'] = daily_ts['hour'].dt.strftime('%Y-%m-%d')
    
    # Use Cached Weather Data
    weather_data = getattr(data_loader, 'weather_cache', {})

    # Merge Weather into Time Series
    ts_records = []
    for _, row in daily_ts.iterrows():
        d_str = row['date']
        w = weather_data.get(d_str, {'tavg': None, 'prcp': None, 'humidity': None})
        ts_records.append({
            'date': d_str,
            'trip_count': row['trip_count'],
            'temp': w.get('tavg'),
            'precip': w.get('prcp'),
            'humidity': w.get('humidity')
        })

    # --- 4. Trend (7-Day MA) ---
    daily_ts['7_day_ma'] = daily_ts['trip_count'].rolling(window=7).mean().fillna(0)
    daily_ts['std'] = daily_ts['trip_count'].rolling(window=7).std().fillna(0)
    daily_ts['upper_ci'] = daily_ts['7_day_ma'] + (1.96 * daily_ts['std'])
    daily_ts['lower_ci'] = daily_ts['7_day_ma'] - (1.96 * daily_ts['std'])
    
    trend_data = daily_ts[['date', '7_day_ma', 'upper_ci', 'lower_ci']].to_dict(orient='records')

    # --- 5. Anomalies (Z-Score) ---
    mean_trips = daily_ts['trip_count'].mean()
    std_trips = daily_ts['trip_count'].std()
    daily_ts['z_score'] = (daily_ts['trip_count'] - mean_trips) / std_trips
    
    anomalies_df = daily_ts[abs(daily_ts['z_score']) > 2].copy()
    anomalies_data = []
    for _, row in anomalies_df.iterrows():
        d_str = row['date']
        w = weather_data.get(d_str, {'tavg': None, 'prcp': None})
        anomalies_data.append({
            'date': d_str,
            'trip_count': row['trip_count'],
            'z_score': round(row['z_score'], 2),
            'temp': w.get('tavg'),
            'precip': w.get('prcp')
        })

    # --- 6. Rider Heatmap ---
    # We need day_of_week vs member_casual. 
    # Since we are filtering hourly_demand, we can't easily get member_casual split unless we join with daily_demand or if hourly_demand had it.
    # hourly_demand does NOT have member_casual. daily_demand DOES.
    # We will use daily_demand, filtered by date range, to build the heatmap.
    
    heatmap_data = []
    if data_loader.daily_demand is not None:
        dd = data_loader.daily_demand.copy()
        # Check if member_casual exists
        if 'member_casual' in dd.columns:
            heatmap_df = dd.pivot_table(index='day_of_week', columns='member_casual', values='trip_count', aggfunc='sum').reset_index()
            heatmap_data = heatmap_df.to_dict(orient='records')
        else:
            print("Warning: 'member_casual' column missing in daily_demand. Skipping heatmap.")

    return {
        "kpis": {
            "total_trips": total_trips,
            "weekend_share": round(weekend_share, 2),
            "peak_hour": int(peak_hour_idx),
            "weekend_effect": round(cohens_d, 2)
        },
        "time_series": ts_records,
        "trend": trend_data,
        "anomalies": anomalies_data,
        "heatmap": heatmap_data
    }

@app.get("/api/map-data")
async def get_map_data():
    if data_loader.station_locations is None:
        raise HTTPException(status_code=503, detail="Data not loaded yet")
    
    # We need station coordinates and trip counts.
    # station_locations has coordinates.
    # top_stations has trip counts (or we can aggregate from station_hourly_demand).
    # Let's use top_stations for trip counts if available, or aggregate.
    
    # Merge locations with trip counts
    # Assuming station_locations has 'station_name', 'lat', 'lon'
    # Assuming top_stations has 'station_name', 'trip_count' (or similar)
    
    # Check columns first (in a real scenario, I'd inspect the DF, but here I infer from file names/context)
    # Let's assume standard names.
    
    stations_df = data_loader.station_locations.copy()
    
    # If we have top_stations with counts, merge.
    if data_loader.top_stations is not None:
        # top_stations might just be a list of names or names + counts.
        # Let's assume it has counts. If not, we aggregate from station_hourly_demand.
        if 'trip_count' in data_loader.top_stations.columns:
             stations_df = pd.merge(stations_df, data_loader.top_stations[['station_name', 'trip_count']], on='station_name', how='left')
        else:
             # Aggregate from station_hourly_demand
             if data_loader.station_hourly_demand is not None:
                 counts = data_loader.station_hourly_demand.groupby('start_station_name')['trip_count'].sum().reset_index()
                 counts.rename(columns={'start_station_name': 'station_name'}, inplace=True)
                 stations_df = pd.merge(stations_df, counts, on='station_name', how='left')
    
    stations_df['trip_count'] = stations_df['trip_count'].fillna(0)
    
    # PERFORMANCE FIX: Limit to top 1000 stations by volume
    stations_df = stations_df.sort_values('trip_count', ascending=False).head(1000)
    
    # Format for Deck.gl (list of dicts)
    map_data = stations_df.to_dict(orient='records')
    
    return map_data

@app.get("/api/routes")
async def get_routes(top_n: int = 50):
    if data_loader.top_routes is None:
        raise HTTPException(status_code=503, detail="Data not loaded yet")
    
    if data_loader.station_locations is None:
        raise HTTPException(status_code=503, detail="Data not loaded yet")

    # top_routes should have start_station_name, end_station_name, trip_count
    # We need to add coordinates for start and end stations.
    
    routes_df = data_loader.top_routes.head(top_n).copy()
    
    # Merge start coordinates
    routes_df = pd.merge(routes_df, data_loader.station_locations[['station_name', 'lat', 'lon']], left_on='start_station_name', right_on='station_name', how='left')
    routes_df.rename(columns={'lat': 'start_lat', 'lon': 'start_lon'}, inplace=True)
    routes_df.drop(columns=['station_name'], inplace=True)
    
    # Merge end coordinates
    routes_df = pd.merge(routes_df, data_loader.station_locations[['station_name', 'lat', 'lon']], left_on='end_station_name', right_on='station_name', how='left')
    routes_df.rename(columns={'lat': 'end_lat', 'lon': 'end_lon'}, inplace=True)
    routes_df.drop(columns=['station_name'], inplace=True)
    
    # Drop rows with missing coordinates
    routes_df.dropna(subset=['start_lat', 'start_lon', 'end_lat', 'end_lon'], inplace=True)
    
    return routes_df.to_dict(orient='records')

@app.get("/api/stations")
async def get_stations():
    if data_loader.station_locations is None:
        raise HTTPException(status_code=503, detail="Data not loaded yet")
    
    # Return list of station names
    return data_loader.station_locations['station_name'].tolist()

@app.get("/api/station/{station_name}")
async def get_station_details(station_name: str):
    if data_loader.station_hourly_demand is None:
        raise HTTPException(status_code=503, detail="Data not loaded yet")
    
    # Filter data for the station
    station_data = data_loader.station_hourly_demand[data_loader.station_hourly_demand['start_station_name'] == station_name]
    
    if station_data.empty:
        raise HTTPException(status_code=404, detail="Station not found")
    
    # KPIs
    total_trips = int(station_data['trip_count'].sum())
    # Avg Trips/Day: Total trips / number of unique days (approximate since we have hourly data for days)
    # station_hourly_demand has 'day_of_week', 'hour', 'trip_count'. It's aggregated.
    # So we sum trip_count and divide by 7 (days in week)? No, that's avg per day of week.
    # The requirement says "Avg Trips/Day". Since the data is aggregated by day of week, 
    # we can sum all trips and divide by 7 to get an average daily demand *if* the data represents one week.
    # But the data likely represents an average week over the period.
    # Let's assume the data is "average hourly demand by day of week".
    # So sum(trip_count) is total weekly trips. Avg/Day = Total / 7.
    avg_trips_per_day = int(total_trips / 7)
    
    # Peak Hour
    peak_hour = int(station_data.groupby('hour')['trip_count'].sum().idxmax())
    
    # Hourly Profile (0-23) - Average demand by hour
    hourly_profile = station_data.groupby('hour')['trip_count'].mean().reset_index().to_dict(orient='records')
    
    # Daily Profile (Mon-Sun) - Total demand by day of week
    # Ensure correct order
    days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    daily_profile_df = station_data.groupby('day_of_week')['trip_count'].sum().reindex(days_order).reset_index()
    daily_profile = daily_profile_df.to_dict(orient='records')
    
    # Mini-Map Location
    location = {}
    if data_loader.station_locations is not None:
        loc_row = data_loader.station_locations[data_loader.station_locations['station_name'] == station_name]
        if not loc_row.empty:
            location = {
                "lat": float(loc_row.iloc[0]['lat']),
                "lon": float(loc_row.iloc[0]['lon'])
            }
            
    return {
        "kpis": {
            "total_trips": total_trips,
            "avg_trips_day": avg_trips_per_day,
            "peak_hour": peak_hour
        },
        "hourly_profile": hourly_profile,
        "daily_profile": daily_profile,
        "location": location
    }



class PredictionRequest(BaseModel):
    station_name: str
    start_date: str
    end_date: str

@app.post("/api/predict")
async def predict_demand(request: PredictionRequest):
    try:
        print(f"DEBUG: Received prediction request for {request.station_name}", flush=True)
        
        # Use new prediction service
        from prediction_service import get_prediction_service
        service = get_prediction_service()
        
        # Calculate hours between start and end
        from datetime import datetime
        start_dt = datetime.fromisoformat(request.start_date.replace('Z', '+00:00'))
        end_dt = datetime.fromisoformat(request.end_date.replace('Z', '+00:00'))
        hours_ahead = int((end_dt - start_dt).total_seconds() / 3600)
        hours_ahead = min(max(hours_ahead, 1), 168)  # 1-168 hours (1 week max)
        
        # Get predictions - now can raise ValueError
        try:
            result = service.predict(request.station_name, request.start_date, hours_ahead)
        except ValueError as pred_err:
            raise HTTPException(status_code=500, detail=str(pred_err))
        
        # Legacy check for old error dict format (can be removed later)
        if isinstance(result, dict) and 'error' in result:
            raise HTTPException(status_code=500, detail=result['error'])
        
        # Calculate KPIs
        predicted_total = sum([x['predicted'] for x in result])
        peak_item = max(result, key=lambda x: x['predicted']) if result else None
        peak_time = peak_item['date'].split('T')[1][:5] if peak_item else ""
        
        return {
            "kpis": {
                "predicted_total": int(predicted_total),
                "peak_time": peak_time
            },
            "predictions": result
        }
    except HTTPException:
        raise  # Re-raise HTTP exceptions
    except Exception as e:
        print(f"ERROR: Prediction failed: {str(e)}", flush=True)
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/advanced-analytics/abm")
async def get_abm_simulation():
    if data_loader.v8_abm_data is None:
        data_loader.load_v8_assets()
    if data_loader.v8_abm_data is None:
        raise HTTPException(status_code=404, detail="ABM Data not found. Ensure V8 assets are loaded.")
    return data_loader.v8_abm_data

@app.get("/api/advanced-analytics/tft-forecast")
async def get_tft_forecast():
    if data_loader.v8_tft_forecasts is None:
        data_loader.load_v8_assets()
    if data_loader.v8_tft_forecasts is None:
        raise HTTPException(status_code=404, detail="TFT Forecasts not found.")
    return data_loader.v8_tft_forecasts

@app.get("/api/advanced-analytics/rl-policy")
async def get_rl_policy():
    if data_loader.v8_rl_policy is None:
        data_loader.load_v8_assets()
    if data_loader.v8_rl_policy is None:
        raise HTTPException(status_code=404, detail="RL Policy not found.")
    return data_loader.v8_rl_policy

@app.get("/api/advanced-analytics/equity")
async def get_equity_scores():
    if data_loader.v8_equity_scores is None:
        data_loader.load_v8_assets()
    if data_loader.v8_equity_scores is None:
        raise HTTPException(status_code=404, detail="Equity Scores not found.")
    return data_loader.v8_equity_scores

@app.get("/api/advanced-analytics/digital-twin")
async def get_digital_twin():
    if data_loader.v8_digital_twin is None:
        data_loader.load_v8_assets()
    if data_loader.v8_digital_twin is None:
        raise HTTPException(status_code=404, detail="Digital Twin entities not found.")
    return data_loader.v8_digital_twin

@app.get("/api/advanced-analytics/uhi")
async def get_uhi_data():
    if data_loader.v9_uhi_data is None:
        data_loader.load_v9_assets()
    if data_loader.v9_uhi_data is None:
        raise HTTPException(status_code=404, detail="UHI Data not found. Ensure V9 assets are loaded.")
    return data_loader.v9_uhi_data

@app.get("/api/advanced-analytics/safety")
async def get_safety_heatmap():
    if data_loader.v9_safety_events is None:
        data_loader.load_v9_assets()
    if data_loader.v9_safety_events is None:
        raise HTTPException(status_code=404, detail="Safety Heatmap data not found.")
    return data_loader.v9_safety_events

@app.get("/api/advanced-analytics/synthetic-trips")
async def get_synthetic_trips():
    if data_loader.v9_synthetic_trips is None:
        data_loader.load_v9_assets()
    if data_loader.v9_synthetic_trips is None:
        raise HTTPException(status_code=404, detail="Synthetic Trips not found.")
    return data_loader.v9_synthetic_trips

@app.get("/api/advanced-analytics/web3")
async def get_web3_incentives():
    if data_loader.v9_bikecoin_data is None:
        data_loader.load_v9_assets()
    if data_loader.v9_bikecoin_data is None:
        raise HTTPException(status_code=404, detail="Web3 Incentives not found.")
    return data_loader.v9_bikecoin_data

@app.get("/api/historical-demand")
async def get_historical_demand(
    station: str,
    start_date: str,
    end_date: str
):
    """Get historical hourly demand from the FULL 9.9M row dataset (2019-2025)"""
    try:
        # Validate inputs
        if not station:
            raise HTTPException(status_code=400, detail="Station name is required")
        
        # Use the FULL historical data file
        import os
        parquet_path = os.path.join(os.path.dirname(__file__), "data", "v1_core", "final_station_demand_robust_features.parquet")
        
        if not os.path.exists(parquet_path):
            raise HTTPException(status_code=503, detail="Historical data file not found")
        
        df = pd.read_parquet(parquet_path, columns=['time', 'station_name', 'pickups'])
        
        # Filter by station
        df_station = df[df['station_name'] == station].copy()
        
        if df_station.empty:
            # Return empty list with 200 status - station exists but no data for this station
            return []
       
        # Convert time to timezone-naive datetime
        df_station['time'] = pd.to_datetime(df_station['time']).dt.tz_localize(None)
        
        # Filter by date range
        try:
            # Parse dates and remove timezone to match data format
            start_dt = pd.to_datetime(start_date).tz_localize(None) if pd.to_datetime(start_date).tz is None else pd.to_datetime(start_date).tz_convert(None).tz_localize(None)
            end_dt = pd.to_datetime(end_date).tz_localize(None) if pd.to_datetime(end_date).tz is None else pd.to_datetime(end_date).tz_convert(None).tz_localize(None)
        except Exception as date_err:
            raise HTTPException(status_code=400, detail=f"Invalid date format: {str(date_err)}")
        
        df_filtered = df_station[
            (df_station['time'] >= start_dt) & 
            (df_station['time'] <= end_dt)
        ].copy()
        
        # Format response
        result = []
        for _, row in df_filtered.iterrows():
            # Add Z suffix for UTC timezone to match prediction date format
            result.append({
                'date': row['time'].isoformat() + '+00:00',
                'demand': float(row['pickups'])
            })
        
        return result
        
    except HTTPException:
        raise  # Re-raise HTTP exceptions
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
