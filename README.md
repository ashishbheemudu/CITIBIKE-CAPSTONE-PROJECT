# Citi Bike Analytics Platform

Real-time bike-sharing analytics and demand forecasting system for NYC Citi Bike.

## 🚀 Features

- **System Overview**: Real-time metrics and KPIs
- **Interactive Map**: Live station status and availability
- **Route Analysis**: Top 200 bike routes visualization
- **Station Drilldown**: Detailed analytics for individual stations
- **Demand Forecaster**: 48-hour demand predictions using ML models
- **Fleet Command**: Rebalancing dashboard with live GBFS data
- **Social Equity**: Accessibility heatmaps for underserved areas

## 📦 Tech Stack

### Backend
- **Framework**: FastAPI (Python)
- **Data Processing**: Pandas, NumPy
- **ML Models**: XGBoost, LightGBM, CatBoost
- **Real-time Data**: GBFS (General Bikeshare Feed Specification)
- **Server**: Uvicorn + Nginx (HTTPS)

### Frontend
- **Framework**: React + Vite
- **Styling**: Tailwind CSS
- **Charts**: Chart.js, Recharts
- **Maps**: Leaflet
- **HTTP Client**: Axios

## 🏗️ Project Structure

```
cap/
├── backend/
│   ├── main.py                 # FastAPI application
│   ├── prediction_service.py   # ML prediction engine
│   ├── data_loader.py          # Data management
│   ├── gbfs_service.py         # Live GBFS integration
│   ├── generate_advanced_data.py
│   ├── citibike.service        # Systemd service config
│   ├── data/
│   │   ├── v1_core/            # Core datasets
│   │   ├── v8_abm/             # Advanced analytics
│   │   └── v9_omniscient/      # Omniscient features
│   └── models/                 # ML models and scalers
│
└── frontend/
    ├── src/
    │   ├── pages/              # Dashboard views
    │   ├── api.js              # API client
    │   └── main.jsx            # App entry point
    ├── public/
    └── vite.config.js
```

## 🔧 Setup

### Prerequisites
- Python 3.9+
- Node.js 18+
- npm or yarn

### Backend Setup

```bash
cd backend

# Install dependencies
pip install fastapi uvicorn pandas numpy scikit-learn requests

# Install ML libraries (optional - fallback predictor works without)
pip install xgboost lightgbm catboost

# Run development server
python3 -m uvicorn main:app --reload --port 8000
```

### Frontend Setup

```bash
cd frontend

# Install dependencies
npm install

# Run development server
npm run dev
```

The app will be available at `http://localhost:3000`

## 🚀 Deployment

### Backend (EC2)

1. **Setup systemd service:**
```bash
sudo cp citibike.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable citibike
sudo systemctl start citibike
```

2. **Check status:**
```bash
sudo systemctl status citibike
```

### Frontend (Vercel)

1. **Connect GitHub repository** to Vercel
2. **Set environment variable:**
   - `VITE_API_URL`: Your backend API URL
3. **Deploy** automatically on push to main

## 📊 API Endpoints

### Core Endpoints
- `GET /api/system-overview` - Dashboard KPIs
- `GET /api/stations` - Station list
- `GET /api/map-data` - Map markers
- `POST /api/predict` - Demand predictions

### Analytics
- `GET /api/routes` - Top bike routes
- `GET /api/stations/{name}/analytics` - Station details
- `GET /api/advanced-analytics/equity` - Equity scores
- `GET /api/rebalancing` - Rebalancing actions

### Live Data
- `GET /api/live/stations` - Real-time GBFS data

## 🧠 ML Model Fallback

The system includes a **statistical fallback predictor** that activates when ML libraries are unavailable:

- Uses historical hourly patterns
- Calculates day-of-week trends
- Generates realistic demand forecasts
- No ML library dependencies required

## 🔒 Environment Variables

### Frontend (.env.production)
```
VITE_API_URL=https://your-backend-url.com/api
```

### Backend
No environment variables required - uses relative paths for data files.

## 🐛 Troubleshooting

### Backend won't start
```bash
# Check service logs
sudo journalctl -u citibike -n 50

# Restart service
sudo systemctl restart citibike
```

### Predictions timing out
The system automatically falls back to statistical prediction if ML models fail. Check logs for:
```
⚠️ Falling back to STATISTICAL predictor
```

### Frontend can't connect to backend
1. Verify `VITE_API_URL` in Vercel settings
2. Check CORS is enabled in `main.py`
3. Test backend directly: `curl https://your-backend/api/stations`

## 📈 Performance

- **Backend startup**: ~3 seconds
- **Prediction generation**: 2-5 seconds (fallback), 5-10 seconds (ML)
- **API response time**: <500ms (average)
- **Data refresh**: 5 minutes (GBFS), hourly (analytics)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test locally
5. Submit a pull request

## 📝 License

MIT License - see LICENSE file for details

## 🔗 Links

- **Live Demo**: https://citibike-capstone-project.vercel.app
- **Backend API**: https://3.22.236.184.nip.io/api
- **GBFS Feed**: https://gbfs.citibikenyc.com/gbfs/gbfs.json

---

Built with ❤️ for NYC bike share analytics
