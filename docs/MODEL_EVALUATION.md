# Model Evaluation Report

## Executive Summary

This document presents the comprehensive evaluation of machine learning models developed for NYC Citi Bike demand prediction.

**Best Model:** 3-Model Ensemble (XGBoost + LightGBM + CatBoost)  
**Performance:** R² = 0.781, MAE = 2.66 trips/hour

---

## 1. Evaluation Methodology

### 1.1 Train-Test Split
- **Training Period:** January 2020 - September 2024 (80%)
- **Testing Period:** October 2024 - November 2025 (20%)
- **Split Type:** Time-based (prevents data leakage)

### 1.2 Cross-Validation
- **Method:** 5-Fold Time Series Split
- **Purpose:** Robust performance estimation

### 1.3 Metrics Used

| Metric | Description | Target |
|--------|-------------|--------|
| R² Score | Variance explained | > 0.75 |
| MAE | Mean Absolute Error | < 3.0 |
| RMSE | Root Mean Squared Error | < 5.0 |
| MAPE | Mean Absolute Percentage Error | < 20% |

---

## 2. Model Comparison

### 2.1 Performance Summary

| Model | R² Score | MAE | RMSE | Training Time |
|-------|----------|-----|------|---------------|
| XGBoost | 0.779 | 2.68 | 4.12 | 45 sec |
| LightGBM | 0.780 | 2.64 | 4.08 | 28 sec |
| CatBoost | 0.778 | 2.69 | 4.14 | 62 sec |
| Random Forest | 0.776 | 2.72 | 4.18 | 180 sec |
| **Ensemble** | **0.781** | **2.66** | **4.10** | - |

### 2.2 Statistical Significance
- All models significantly outperform baseline (p < 0.001)
- Ensemble marginally better than individual models

---

## 3. Feature Importance Analysis

### 3.1 Top 15 Features (Ensemble)

| Rank | Feature | Importance | Category |
|------|---------|------------|----------|
| 1 | demand_1h | 0.142 | Lag |
| 2 | hour | 0.098 | Temporal |
| 3 | demand_24h | 0.087 | Lag |
| 4 | rolling_mean_3h | 0.076 | Lag |
| 5 | station_avg_hourly | 0.068 | Station |
| 6 | is_weekend | 0.054 | Temporal |
| 7 | temp | 0.052 | Weather |
| 8 | demand_168h | 0.048 | Lag |
| 9 | is_rush_hour | 0.045 | Temporal |
| 10 | month | 0.041 | Temporal |
| 11 | rolling_std_24h | 0.038 | Lag |
| 12 | humidity | 0.035 | Weather |
| 13 | dow_sin | 0.032 | Temporal |
| 14 | demand_6h | 0.029 | Lag |
| 15 | ema_24h | 0.027 | Lag |

### 3.2 Feature Category Contribution

| Category | Total Importance | Features |
|----------|-----------------|----------|
| Lag Features | 44.7% | 18 |
| Temporal Features | 32.1% | 19 |
| Station Features | 15.8% | 13 |
| Weather Features | 7.4% | 4 |

---

## 4. Error Analysis

### 4.1 Error Distribution
- Mean Error: -0.12 (slight underestimation)
- Error Std Dev: 4.08
- 90% of predictions within ±7 trips

### 4.2 Error by Time of Day

| Time Period | MAE | Notes |
|-------------|-----|-------|
| Night (0-6) | 1.82 | Best performance |
| Morning Rush (7-9) | 3.45 | Higher error |
| Midday (10-16) | 2.54 | Good performance |
| Evening Rush (17-19) | 3.89 | Highest error |
| Evening (20-23) | 2.21 | Good performance |

### 4.3 Error by Day Type

| Day Type | MAE | R² |
|----------|-----|-----|
| Weekday | 2.89 | 0.768 |
| Weekend | 2.31 | 0.812 |

---

## 5. Calibration Analysis

### 5.1 Prediction vs Actual
- Correlation: 0.884
- Slope: 0.96 (slight underestimation of peaks)
- Intercept: 0.42

### 5.2 Calibration Applied
To correct systematic underestimation:
```python
calibration_factor = 3.0  # Based on historical variance
calibrated = base_prediction + (base_prediction - mean) * calibration_factor
```

---

## 6. Model Robustness

### 6.1 Temporal Stability
- Performance consistent across months
- No significant drift observed

### 6.2 Station Generalization
- Models generalize well to stations with >1000 historical trips
- Lower accuracy for recently added stations

---

## 7. Comparison with Previous Version

| Metric | v2 (Old) | v3 (Current) | Improvement |
|--------|----------|--------------|-------------|
| R² Score | 0.264 | 0.781 | +196% |
| MAE | 5.12 | 2.66 | -48% |
| Features | 56 | 54 | Reduced leakage |

---

## 8. Conclusion

The ensemble model achieves production-ready performance with:
- **78.1% variance explained**
- **2.66 trips/hour average error**
- **Robust across stations and time periods**

### Recommendations
1. Deploy ensemble for production predictions
2. Implement weekly model monitoring
3. Retrain quarterly with new data
4. Add confidence intervals to predictions
