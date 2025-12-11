# Data Dictionary

## Overview

This document describes all data sources, features, and data processing pipelines used in the NYC Citi Bike Analytics Dashboard.

---

## Data Sources

### 1. Citi Bike Trip Data

| Field | Type | Description |
|-------|------|-------------|
| `started_at` | datetime | Trip start timestamp |
| `ended_at` | datetime | Trip end timestamp |
| `start_station_name` | string | Origin station name |
| `end_station_name` | string | Destination station name |
| `start_lat` | float | Origin latitude |
| `start_lng` | float | Origin longitude |
| `end_lat` | float | Destination latitude |
| `end_lng` | float | Destination longitude |
| `member_casual` | string | User type (member/casual) |
| `rideable_type` | string | Bike type (classic/electric/docked) |

**Source:** NYC Citi Bike System Data (https://citibikenyc.com/system-data)  
**Time Range:** January 2020 - November 2025  
**Total Records:** ~68 million trips

---

### 2. Weather Data

| Field | Type | Description |
|-------|------|-------------|
| `date` | date | Date of observation |
| `temp` | float | Temperature in Celsius |
| `precip` | float | Precipitation in mm |
| `humidity` | float | Relative humidity (%) |
| `wind_speed` | float | Wind speed (km/h) |

**Source:** Meteostat API (Central Park weather station)  
**Coverage:** 2020-2025

---

## Derived Features

### Temporal Features (13 features)

| Feature | Type | Description |
|---------|------|-------------|
| `hour` | int | Hour of day (0-23) |
| `day_of_week` | int | Day of week (0=Monday, 6=Sunday) |
| `day_of_month` | int | Day of month (1-31) |
| `month` | int | Month (1-12) |
| `is_weekend` | bool | Saturday or Sunday |
| `is_holiday` | bool | US federal holiday |
| `is_rush_hour` | bool | 7-9 AM or 5-7 PM |
| `hour_sin` | float | Cyclical hour encoding (sin) |
| `hour_cos` | float | Cyclical hour encoding (cos) |
| `dow_sin` | float | Cyclical day of week (sin) |
| `dow_cos` | float | Cyclical day of week (cos) |
| `month_sin` | float | Cyclical month (sin) |
| `month_cos` | float | Cyclical month (cos) |

### Weather Features (4 features)

| Feature | Type | Description |
|---------|------|-------------|
| `temp` | float | Temperature (Celsius) |
| `precip` | float | Precipitation (mm) |
| `humidity` | float | Humidity (%) |
| `wind` | float | Wind speed (if available) |

### Lag Features (24 features)

| Feature | Type | Description |
|---------|------|-------------|
| `demand_1h` | float | Demand 1 hour ago |
| `demand_2h` | float | Demand 2 hours ago |
| `demand_3h` | float | Demand 3 hours ago |
| `demand_6h` | float | Demand 6 hours ago |
| `demand_12h` | float | Demand 12 hours ago |
| `demand_24h` | float | Demand 24 hours ago |
| `demand_168h` | float | Demand 1 week ago |
| `rolling_mean_3h` | float | 3-hour rolling mean |
| `rolling_mean_6h` | float | 6-hour rolling mean |
| `rolling_mean_24h` | float | 24-hour rolling mean |
| `rolling_std_3h` | float | 3-hour rolling std |
| `rolling_std_24h` | float | 24-hour rolling std |
| `ema_24h` | float | 24-hour exponential moving average |
| ... | ... | (additional lag features) |

### Station Features (13 features)

| Feature | Type | Description |
|---------|------|-------------|
| `station_avg_hourly` | float | Historical avg for this station/hour |
| `station_avg_dow` | float | Historical avg for this station/day |
| `station_popularity` | float | Relative popularity score |
| ... | ... | (station-specific aggregations) |

---

## Data Processing Pipeline

```
Raw Trip Data (CSV)
        ↓
    Cleaning
    - Remove null stations
    - Filter outlier durations
        ↓
    Aggregation
    - Hourly trip counts per station
        ↓
    Feature Engineering
    - Add temporal features
    - Add weather features
    - Add lag features
        ↓
    Scaling
    - StandardScaler for tree models
        ↓
    Train/Test Split
    - 80% training (time-based)
    - 20% testing
        ↓
    Model Training
```

---

## Target Variable

| Name | Type | Description |
|------|------|-------------|
| `trip_count` | int | Number of trips starting from a station in an hour |

---

## Data Quality

| Check | Status |
|-------|--------|
| Missing values handled | ✅ |
| Outliers filtered | ✅ |
| Time-based split (no leakage) | ✅ |
| Feature normalization | ✅ |
| Consistent datetime format | ✅ |
