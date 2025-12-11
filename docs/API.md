# API Documentation

## Base URL

- **Production:** `https://18.218.154.66.nip.io/api`
- **Local:** `http://localhost:8000/api`

---

## Authentication

Currently, the API does not require authentication. CORS is configured to allow requests from whitelisted origins only.

---

## Endpoints

### Health Check

#### `GET /`
Returns server health status.

**Response:**
```json
{
  "message": "Backend is running",
  "version": "3.0.0"
}
```

---

### System Overview

#### `GET /api/system-overview`
Get system-wide analytics and KPIs.

**Query Parameters:**
| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `start_date` | string | No | Filter start date (YYYY-MM-DD) |
| `end_date` | string | No | Filter end date (YYYY-MM-DD) |
| `days_of_week` | string | No | Comma-separated days (Monday,Tuesday,...) |

**Response:**
```json
{
  "kpis": {
    "total_trips": 68727428,
    "weekend_share": 26.22,
    "peak_hour": 17,
    "weekend_effect": -0.28
  },
  "time_series": [...],
  "trend": [...],
  "anomalies": [...],
  "heatmap": [...]
}
```

---

### Stations

#### `GET /api/stations`
Get list of all station names.

**Response:**
```json
["1 Ave & E 16 St", "1 Ave & E 18 St", ...]
```

#### `GET /api/station/{station_name}`
Get detailed analytics for a specific station.

**Path Parameters:**
| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `station_name` | string | Yes | URL-encoded station name |

**Response:**
```json
{
  "kpis": {
    "total_trips": 1748,
    "avg_trips_day": 249,
    "peak_hour": 17
  },
  "hourly_profile": [...]
}
```

---

### Map Data

#### `GET /api/map-data`
Get station locations with coordinates and trip counts.

**Response:**
```json
[
  {
    "station_name": "W 21 St & 6 Ave",
    "lat": 40.74173969,
    "lon": -73.99415556,
    "trip_count": 759609
  },
  ...
]
```

---

### Routes

#### `GET /api/routes`
Get top routes between stations.

**Query Parameters:**
| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `top_n` | int | No | 50 | Number of routes to return |

**Response:**
```json
[
  {
    "start_station_name": "W 21 St & 6 Ave",
    "end_station_name": "West St & Chambers St",
    "trip_count": 141177,
    "start_lat": 40.74173969,
    "start_lon": -73.99415556,
    "end_lat": 40.717683315,
    "end_lon": -74.013174415
  },
  ...
]
```

---

### Predictions (Neural Demand)

#### `POST /api/predict`
Generate bike demand predictions using ML ensemble.

**Request Body:**
```json
{
  "station_name": "Broadway & W 58 St",
  "start_date": "2025-12-11T00:00:00",
  "end_date": "2025-12-11T12:00:00"
}
```

**Response:**
```json
{
  "predictions": [
    {
      "date": "2025-12-11T00:00:00",
      "predicted": 8.400051838715576
    },
    ...
  ]
}
```

---

### Historical Demand

#### `GET /api/historical-demand`
Get historical demand data for comparison with predictions.

**Query Parameters:**
| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `station` | string | Yes | Station name (URL-encoded) |
| `start_date` | string | Yes | Start date (YYYY-MM-DD) |
| `end_date` | string | Yes | End date (YYYY-MM-DD) |

**Response:**
```json
[
  {
    "date": "2024-01-01T00:00:00+00:00",
    "demand": 8.0
  },
  ...
]
```

---

### Advanced Analytics

#### `GET /api/advanced-analytics/abm`
Agent-Based Model simulation data.

#### `GET /api/advanced-analytics/tft-forecast`
Temporal Fusion Transformer forecasts.

#### `GET /api/advanced-analytics/rl-policy`
Reinforcement Learning rebalancing policies.

#### `GET /api/advanced-analytics/equity`
Station equity and accessibility scores.

#### `GET /api/advanced-analytics/digital-twin`
Digital twin simulation entities.

#### `GET /api/advanced-analytics/uhi`
Urban Heat Island temperature data.

#### `GET /api/advanced-analytics/safety`
Safety event heatmap data.

#### `GET /api/advanced-analytics/synthetic-trips`
Synthetic trip generation data.

#### `GET /api/advanced-analytics/web3`
Web3/BikeCoin incentive data.

---

## Error Handling

All errors return a JSON response with the following structure:

```json
{
  "detail": "Error message describing what went wrong"
}
```

### HTTP Status Codes

| Code | Description |
|------|-------------|
| 200 | Success |
| 400 | Bad Request - Invalid parameters |
| 404 | Not Found - Resource doesn't exist |
| 422 | Validation Error - Invalid input format |
| 500 | Internal Server Error |
| 503 | Service Unavailable - Data not loaded |

---

## Rate Limiting

Currently no rate limiting is implemented. For production use, consider implementing request throttling.
