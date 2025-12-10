
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
