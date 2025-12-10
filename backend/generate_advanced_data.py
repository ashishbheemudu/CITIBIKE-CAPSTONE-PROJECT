
import os
import json
import random
import numpy as np

# Setup paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
V8_DIR = os.path.join(DATA_DIR, 'v8_abm')
V9_DIR = os.path.join(DATA_DIR, 'v9_omniscient')

os.makedirs(V8_DIR, exist_ok=True)
os.makedirs(V9_DIR, exist_ok=True)

# NYC Bounding Box
NYC_LAT_MIN, NYC_LAT_MAX = 40.7000, 40.8000
NYC_LON_MIN, NYC_LON_MAX = -74.0200, -73.9300

def random_loc():
    return {
        "lat": random.uniform(NYC_LAT_MIN, NYC_LAT_MAX),
        "lng": random.uniform(NYC_LON_MIN, NYC_LON_MAX)
    }

# 1. ABM Simulation Data
print("Generating ABM Simulation Data...")
agents = []
for i in range(50):
    loc = random_loc()
    agents.append({
        "agent_id": f"AG-{i:03d}",
        "lat": loc["lat"],
        "lng": loc["lng"],
        "state": random.choice(["RIDING", "IDLE", "CHARGING"]),
        "energy_level": round(random.uniform(20, 100), 1),
        "decision_factor": random.choice(["Demand", "Weather", "Rebalance"]),
        "decision_target": f"Station-{random.randint(100, 999)}"
    })
with open(os.path.join(V8_DIR, 'abm_simulation.json'), 'w') as f:
    json.dump(agents, f)

# 2. UHI (Urban Heat Island) Data
print("Generating UHI Data...")
uhi_points = []
for i in range(200):
    loc = random_loc()
    # Create hotspots
    is_hotspot = random.random() < 0.2
    intensity = random.uniform(0.7, 1.0) if is_hotspot else random.uniform(0.1, 0.4)
    uhi_points.append({
        "lat": loc["lat"],
        "lng": loc["lng"],
        "heat_stress_index": round(intensity, 2)
    })
with open(os.path.join(V9_DIR, 'uhi_analysis.json'), 'w') as f:
    json.dump(uhi_points, f)

# 3. Safety Events
print("Generating Safety Data...")
safety_events = []
event_types = ["Near Miss", "Pothole", "Intersection Risk", "Dooring Zone"]
for i in range(30):
    loc = random_loc()
    safety_events.append({
        "lat": loc["lat"],
        "lng": loc["lng"],
        "event_type": random.choice(event_types),
        "severity": round(random.uniform(0.4, 0.95), 2)
    })
with open(os.path.join(V9_DIR, 'safety_hotspots.json'), 'w') as f:
    json.dump(safety_events, f)

# 4. Web3 Incentives
print("Generating Web3 Data...")
web3_nodes = []
for i in range(40):
    loc = random_loc()
    web3_nodes.append({
        "lat": loc["lat"],
        "lng": loc["lng"],
        "action_type": random.choice(["pickup", "dropoff"]),
        "bikecoin_reward": round(random.uniform(0.5, 5.0), 2)
    })
with open(os.path.join(V9_DIR, 'web3_incentives.json'), 'w') as f:
    json.dump(web3_nodes, f)

print("✅ Advanced Analytics Data Generation Complete.")

# 5. TFT Forecasts (V8)
print("Generating TFT Forecasts...")
tft_data = []
for i in range(24):
    tft_data.append({
        "hour": i,
        "predicted": round(random.uniform(500, 2000), 1),
        "actual": round(random.uniform(480, 2050), 1),
        "confidence_lower": round(random.uniform(400, 480), 1),
        "confidence_upper": round(random.uniform(2050, 2200), 1)
    })
with open(os.path.join(V8_DIR, 'tft_forecasts.json'), 'w') as f:
    json.dump(tft_data, f)

# 6. RL Policy (V8)
print("Generating RL Policy...")
rl_data = []
for i in range(10):
    rl_data.append({
        "station_id": f"ST-{random.randint(100,999)}",
        "action": random.choice(["ADD_BIKES", "REMOVE_BIKES", "DO_NOTHING"]),
        "confidence": round(random.random(), 2),
        "reward_expected": round(random.uniform(1.0, 10.0), 2)
    })
with open(os.path.join(V8_DIR, 'rl_policy.json'), 'w') as f:
    json.dump(rl_data, f)

# 7. Equity Scores (V8)
print("Generating Equity Scores...")
equity_data = []
for i in range(50):
    loc = random_loc()
    equity_data.append({
        "lat": loc["lat"],
        "lng": loc["lng"],
        "accessibility_score": round(random.uniform(0.3, 0.95), 2),
        "demographic_index": round(random.uniform(0.1, 1.0), 2)
    })
with open(os.path.join(V8_DIR, 'equity_scores.json'), 'w') as f:
    json.dump(equity_data, f)

# 8. Digital Twin (V8)
print("Generating Digital Twin Entities...")
dt_data = []
for i in range(20):
    loc = random_loc()
    dt_data.append({
        "entity_id": f"DT-{i:03d}",
        "type": random.choice(["STATION", "BIKE", "HUB"]),
        "lat": loc["lat"],
        "lng": loc["lng"],
        "status": "ONLINE",
        "last_sync": "2025-12-09T12:00:00Z"
    })
with open(os.path.join(V8_DIR, 'digital_twin.json'), 'w') as f:
    json.dump(dt_data, f)

# 9. Synthetic Trips (V9)
print("Generating Synthetic Trips...")
syn_trips = []
for i in range(100):
    start = random_loc()
    end = random_loc()
    syn_trips.append({
        "start_lat": start["lat"],
        "start_lng": start["lng"],
        "end_lat": end["lat"],
        "end_lng": end["lng"],
        "duration": random.randint(300, 3600),
        "mode": "SYNTHETIC_GAN"
    })
with open(os.path.join(V9_DIR, 'synthetic_trips.json'), 'w') as f:
    json.dump(syn_trips, f)

# 10. Anomalies CSV (V1 Core)
# Need to create this in v1_core to avoid missing file warning
V1_DIR = os.path.join(DATA_DIR, 'v1_core')
os.makedirs(V1_DIR, exist_ok=True)
print("Generating Anomalies CSV...")
import pandas as pd
dates = pd.date_range(end=pd.Timestamp.now(), periods=20, freq='D')
anomalies = []
for d in dates:
    anomalies.append({
        "date": d.strftime('%Y-%m-%d'),
        "trip_count": random.randint(1000, 5000),
        "z_score": round(random.uniform(2.0, 4.5), 2),
        "severity": random.choice(["HIGH", "MEDIUM"])
    })
pd.DataFrame(anomalies).to_csv(os.path.join(V1_DIR, 'anomalies.csv'), index=False)

print("✅ ALL Data Generation Complete.")
