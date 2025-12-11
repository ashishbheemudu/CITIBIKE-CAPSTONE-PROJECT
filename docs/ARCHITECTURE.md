# System Architecture

## Overview

The NYC Citi Bike Analytics Dashboard is a full-stack application for bike demand prediction and analytics visualization.

---

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              USER INTERFACE                                  │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                     React Frontend (Vercel)                          │   │
│  │  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────────┐   │   │
│  │  │ System  │ │   Map   │ │ Routes  │ │ Station │ │   Neural    │   │   │
│  │  │Overview │ │Explorer │ │Analysis │ │Drilldown│ │   Demand    │   │   │
│  │  └────┬────┘ └────┬────┘ └────┬────┘ └────┬────┘ └──────┬──────┘   │   │
│  │       └───────────┴───────────┴───────────┴─────────────┘          │   │
│  │                              │                                      │   │
│  │                         API Client                                  │   │
│  └──────────────────────────────┼──────────────────────────────────────┘   │
│                                 │ HTTPS                                     │
└─────────────────────────────────┼───────────────────────────────────────────┘
                                  │
┌─────────────────────────────────┼───────────────────────────────────────────┐
│                                 ▼                                            │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │                   FastAPI Backend (AWS EC2)                           │   │
│  │                                                                       │   │
│  │  ┌──────────────┐    ┌──────────────┐    ┌──────────────────────┐   │   │
│  │  │    main.py   │───▶│ data_loader  │───▶│    CSV/Parquet       │   │   │
│  │  │  (17 APIs)   │    │              │    │      Data            │   │   │
│  │  └──────┬───────┘    └──────────────┘    └──────────────────────┘   │   │
│  │         │                                                            │   │
│  │         ▼                                                            │   │
│  │  ┌──────────────────────────────────────────────────────────────┐   │   │
│  │  │              Prediction Service                               │   │   │
│  │  │  ┌──────────┐  ┌──────────┐  ┌──────────┐                   │   │   │
│  │  │  │ XGBoost  │  │ LightGBM │  │ CatBoost │                   │   │   │
│  │  │  │  Model   │  │  Model   │  │  Model   │                   │   │   │
│  │  │  └────┬─────┘  └────┬─────┘  └────┬─────┘                   │   │   │
│  │  │       └──────────────┼──────────────┘                        │   │   │
│  │  │                      ▼                                        │   │   │
│  │  │              Ensemble Engine                                  │   │   │
│  │  │              (Weighted Avg)                                   │   │   │
│  │  └──────────────────────────────────────────────────────────────┘   │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
│                              AWS EC2 (t3.micro)                              │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

## Component Details

### Frontend (React + Vite)

| Component | Purpose | Key Libraries |
|-----------|---------|---------------|
| `SystemOverview.jsx` | Dashboard KPIs and time series | Recharts |
| `MapExplorer.jsx` | Station map visualization | Leaflet |
| `RouteAnalysis.jsx` | Route flow analysis | Recharts |
| `StationDrilldown.jsx` | Individual station analytics | Recharts |
| `Prediction.jsx` | Neural demand predictions | Recharts |
| `AdvancedAnalytics.jsx` | ABM, TFT, RL features | Recharts |

### Backend (FastAPI)

| Module | Purpose | Key Functions |
|--------|---------|---------------|
| `main.py` | API endpoints (17 total) | All route handlers |
| `data_loader.py` | Data loading and caching | `load_core_data()` |
| `prediction_service.py` | ML prediction engine | `predict()`, `_create_features_batch()` |

### ML Models

| Model | R² Score | File Size | Inference Time |
|-------|----------|-----------|----------------|
| XGBoost | 0.78 | 59 MB | ~10ms |
| LightGBM | 0.78 | 11 MB | ~5ms |
| CatBoost | 0.78 | 2.4 MB | ~8ms |
| **Ensemble** | **0.781** | - | ~35ms |

### Data Flow

1. **User Request** → Frontend sends API request
2. **API Handler** → FastAPI routes request to handler
3. **Data Loading** → Lazy-load historical data if needed
4. **Feature Engineering** → Create 54 features for prediction
5. **Model Inference** → Run through ensemble models
6. **Response** → Return predictions to frontend

---

## Deployment Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│     GitHub      │───▶│  GitHub Actions │───▶│   AWS EC2       │
│   Repository    │    │    (CI/CD)      │    │   Backend       │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                                             │
         │                                             │
         ▼                                             ▼
┌─────────────────┐                           ┌─────────────────┐
│     Vercel      │◀──────────────────────────│    Nginx        │
│    Frontend     │        HTTPS              │   (Reverse      │
└─────────────────┘                           │    Proxy)       │
                                              └─────────────────┘
```

---

## Technology Stack

### Backend
- **Framework:** FastAPI 0.109.2
- **Server:** Uvicorn
- **ML:** XGBoost, LightGBM, CatBoost, scikit-learn
- **Data:** Pandas, NumPy

### Frontend
- **Framework:** React 18
- **Build:** Vite
- **Charts:** Recharts
- **Maps:** React-Leaflet
- **Styling:** Tailwind CSS

### Infrastructure
- **Backend Hosting:** AWS EC2 (t3.micro)
- **Frontend Hosting:** Vercel
- **CI/CD:** GitHub Actions
- **SSL:** Let's Encrypt (via nip.io)

---

## Security Measures

1. **CORS Whitelist:** Only specific origins allowed
2. **Input Validation:** Pydantic models for all inputs
3. **HTTPS:** TLS encryption in production
4. **No SQL:** File-based data (no injection risk)
