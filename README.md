# Citi Bike Analytics Platform

Production-ready bike-sharing analytics and demand forecasting for NYC Citi Bike.

## Features

- Real-time system metrics and KPIs
- Interactive station map with live availability
- Route analysis (top 200 routes)
- Station-level demand analytics
- **48-hour demand forecasting** with ML models + statistical fallback
- Fleet rebalancing dashboard
- Social equity accessibility heatmaps

## Tech Stack

**Backend**: FastAPI, Pandas, NumPy, XGBoost/LightGBM/CatBoost  
**Frontend**: React + Vite, Tailwind CSS, Chart.js, Leaflet  
**Deployment**: AWS EC2 (backend), Vercel (frontend)

## Quick Start

### Backend
```bash
cd backend
pip install -r requirements.txt
python3 -m uvicorn main:app --reload --port 8000
```

### Frontend
```bash
cd frontend
npm install
npm run dev
```

Visit `http://localhost:3000`

## Deployment

### EC2 Backend
```bash
sudo cp citibike.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable citibike
sudo systemctl start citibike
```

### Vercel Frontend
Set environment variable: `VITE_API_URL=https://your-backend-url/api`

## API Endpoints

- `GET /api/system-overview` - Dashboard KPIs
- `GET /api/stations` - Station list
- `POST /api/predict` - Demand predictions (48h)
- `GET /api/routes` - Top bike routes
- `GET /api/rebalancing` - Fleet optimization
- `GET /api/advanced-analytics/equity` - Equity scores

## ML Fallback System

The prediction service includes automatic fallback to statistical methods when ML libraries are unavailable (e.g., GLIBCXX errors on older systems). This ensures predictions always work.

## Links

- **Live Demo**: https://citibike-capstone-project.vercel.app
- **Backend API**: https://3.22.236.184.nip.io/api

---

Built for NYC bike share analytics
