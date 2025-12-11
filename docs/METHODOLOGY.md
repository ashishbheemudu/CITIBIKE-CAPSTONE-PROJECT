# Methodology

## 1. Problem Statement

Predict hourly bike demand at NYC Citi Bike stations to enable proactive bike rebalancing and improve user experience.

**Research Question:** Can we accurately predict the number of bikes needed at each station for the next 48 hours using historical trip data, weather information, and temporal patterns?

---

## 2. Data Collection

### 2.1 Primary Data Source
- **Source:** NYC Citi Bike System Data
- **Format:** CSV files (monthly releases)
- **Time Range:** January 2020 - November 2025
- **Volume:** ~68 million trip records

### 2.2 Secondary Data Source
- **Source:** Meteostat API
- **Variables:** Temperature, precipitation, humidity
- **Resolution:** Daily observations
- **Location:** Central Park weather station (NYC)

---

## 3. Data Preprocessing

### 3.1 Data Cleaning
1. Remove records with null station names
2. Filter trips with duration < 1 minute (station tests)
3. Filter trips with duration > 24 hours (lost bikes)
4. Standardize station name formatting

### 3.2 Aggregation
- Aggregate trips by station and hour
- Calculate hourly trip counts as target variable

### 3.3 Feature Engineering
We engineered 54 features across 4 categories:

| Category | Count | Examples |
|----------|-------|----------|
| Temporal | 19 | hour, day_of_week, is_holiday, cyclical encodings |
| Weather | 4 | temperature, precipitation, humidity |
| Lag | 18 | demand_1h, demand_24h, rolling_mean_3h |
| Station | 13 | station_avg_hourly, popularity_score |

### 3.4 Data Splitting
- **Strategy:** Time-based split (no data leakage)
- **Training:** 80% (earlier dates)
- **Testing:** 20% (later dates)

---

## 4. Model Selection

### 4.1 Candidate Models
We evaluated the following algorithms:

| Model | Type | Hyperparameters |
|-------|------|-----------------|
| XGBoost | Gradient Boosting | max_depth=8, n_estimators=500, learning_rate=0.05 |
| LightGBM | Gradient Boosting | num_leaves=64, n_estimators=500, learning_rate=0.05 |
| CatBoost | Gradient Boosting | depth=8, iterations=500, learning_rate=0.05 |
| Random Forest | Ensemble | max_depth=20, n_estimators=200 |

### 4.2 Model Training
- Cross-validation: 5-fold time-series split
- Early stopping: 50 rounds without improvement
- Feature scaling: StandardScaler for consistency

### 4.3 Ensemble Method
Final prediction uses weighted average of top 3 models:
- XGBoost: 24.8%
- LightGBM: 25.2%
- CatBoost: 25.1%
- (RandomForest excluded due to file size)

---

## 5. Evaluation Metrics

| Metric | Formula | Purpose |
|--------|---------|---------|
| **R² Score** | 1 - (SS_res / SS_tot) | Overall variance explained |
| **MAE** | mean(|y - ŷ|) | Average prediction error |
| **RMSE** | sqrt(mean((y - ŷ)²)) | Penalizes large errors |
| **MAPE** | mean(|y - ŷ| / y) * 100 | Percentage error |

---

## 6. Results

### 6.1 Model Performance

| Model | R² | MAE | RMSE |
|-------|-----|-----|------|
| XGBoost | 0.779 | 2.68 | 4.12 |
| LightGBM | 0.780 | 2.64 | 4.08 |
| CatBoost | 0.778 | 2.69 | 4.14 |
| **Ensemble** | **0.781** | **2.66** | **4.10** |

### 6.2 Feature Importance (Top 10)

1. `demand_1h` - Previous hour demand
2. `hour` - Hour of day
3. `demand_24h` - Same hour yesterday
4. `rolling_mean_3h` - Recent trend
5. `station_avg_hourly` - Station baseline
6. `is_weekend` - Weekend indicator
7. `temp` - Temperature
8. `demand_168h` - Same hour last week
9. `is_rush_hour` - Rush hour flag
10. `month` - Month of year

---

## 7. Implementation

### 7.1 System Architecture
- **Backend:** FastAPI (Python)
- **Frontend:** React (Vite)
- **Deployment:** AWS EC2 + Vercel

### 7.2 Prediction Pipeline
1. User selects station and date range
2. Backend generates 54 features
3. Features scaled using pre-trained scaler
4. Ensemble models generate predictions
5. Predictions calibrated based on historical variance
6. Results returned to frontend

---

## 8. Limitations

1. **Data Recency:** Models trained on historical data may not capture recent pattern changes
2. **Extreme Weather:** Poor performance during extreme weather events not in training data
3. **Special Events:** Cannot predict demand during unprecedented events (concerts, emergencies)
4. **Station Changes:** New stations lack historical data for accurate predictions

---

## 9. Future Work

1. Add real-time weather forecasts
2. Implement continuous model retraining
3. Add event calendar integration
4. Develop station-specific models for high-traffic locations

---

## 10. References

1. Chen, T., & Guestrin, C. (2016). XGBoost: A Scalable Tree Boosting System. *Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining*, 785-794. https://doi.org/10.1145/2939672.2939785

2. Ke, G., Meng, Q., Finley, T., Wang, T., Chen, W., Ma, W., Ye, Q., & Liu, T. Y. (2017). LightGBM: A Highly Efficient Gradient Boosting Decision Tree. *Advances in Neural Information Processing Systems*, 30, 3146-3154.

3. Prokhorenkova, L., Gusev, G., Vorobev, A., Dorogush, A. V., & Gulin, A. (2018). CatBoost: Unbiased Boosting with Categorical Features. *Advances in Neural Information Processing Systems*, 31.

4. NYC Citi Bike. (2024). System Data. Retrieved from https://citibikenyc.com/system-data

5. Meteostat. (2024). Weather Data API. Retrieved from https://meteostat.net

6. Breiman, L. (2001). Random Forests. *Machine Learning*, 45(1), 5-32. https://doi.org/10.1023/A:1010933404324

7. Hastie, T., Tibshirani, R., & Friedman, J. (2009). *The Elements of Statistical Learning: Data Mining, Inference, and Prediction* (2nd ed.). Springer.

8. Pedregosa, F., et al. (2011). Scikit-learn: Machine Learning in Python. *Journal of Machine Learning Research*, 12, 2825-2830.

9. O'Brien, O., Cheshire, J., & Batty, M. (2014). Mining bicycle sharing data for generating insights into sustainable transport systems. *Journal of Transport Geography*, 34, 262-273.

10. Froehlich, J., Neumann, J., & Oliver, N. (2009). Sensing and predicting the pulse of the city through shared bicycling. *IJCAI*, 1420-1426.
