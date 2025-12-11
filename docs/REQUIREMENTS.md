# Project Requirements

## 1. Functional Requirements

### 1.1 Core Features

| ID | Requirement | Priority | Status |
|----|-------------|----------|--------|
| FR-01 | Display system-wide analytics dashboard | High | ✅ |
| FR-02 | Interactive map showing station locations | High | ✅ |
| FR-03 | Route analysis between stations | Medium | ✅ |
| FR-04 | Individual station detail view | Medium | ✅ |
| FR-05 | 48-hour demand predictions | High | ✅ |
| FR-06 | Historical data comparison | Medium | ✅ |
| FR-07 | Date range filtering | Medium | ✅ |
| FR-08 | Day of week filtering | Low | ✅ |

### 1.2 Advanced Analytics

| ID | Requirement | Priority | Status |
|----|-------------|----------|--------|
| FR-09 | Agent-Based Model simulation | Low | ✅ |
| FR-10 | Temporal Fusion Transformer forecasts | Low | ✅ |
| FR-11 | Reinforcement Learning policies | Low | ✅ |
| FR-12 | Equity analysis heatmap | Low | ✅ |
| FR-13 | Digital Twin visualization | Low | ✅ |

---

## 2. Non-Functional Requirements

### 2.1 Performance

| ID | Requirement | Target | Actual |
|----|-------------|--------|--------|
| NFR-01 | API response time | < 2s | ✅ ~1.5s |
| NFR-02 | Page load time | < 3s | ✅ ~2s |
| NFR-03 | Prediction generation | < 5s | ✅ ~2s |
| NFR-04 | Support concurrent users | 100+ | ✅ |

### 2.2 Reliability

| ID | Requirement | Target | Actual |
|----|-------------|--------|--------|
| NFR-05 | API uptime | 99% | ✅ |
| NFR-06 | Error handling | Graceful | ✅ |
| NFR-07 | Data validation | All inputs | ✅ |

### 2.3 Security

| ID | Requirement | Target | Actual |
|----|-------------|--------|--------|
| NFR-08 | HTTPS encryption | Required | ✅ |
| NFR-09 | CORS protection | Whitelist | ✅ |
| NFR-10 | Input sanitization | All inputs | ✅ |

### 2.4 Usability

| ID | Requirement | Target | Actual |
|----|-------------|--------|--------|
| NFR-11 | Mobile responsive | Required | ✅ |
| NFR-12 | Loading indicators | All async | ✅ |
| NFR-13 | Error messages | User-friendly | ✅ |

---

## 3. Technical Requirements

### 3.1 Backend

| Component | Requirement | Implementation |
|-----------|-------------|----------------|
| Language | Python 3.11+ | ✅ |
| Framework | FastAPI | ✅ |
| ML Libraries | XGBoost, LightGBM, CatBoost | ✅ |
| Data Processing | Pandas, NumPy | ✅ |

### 3.2 Frontend

| Component | Requirement | Implementation |
|-----------|-------------|----------------|
| Framework | React 18+ | ✅ |
| Build Tool | Vite | ✅ |
| Charts | Recharts | ✅ |
| Maps | Leaflet | ✅ |
| Styling | Tailwind CSS | ✅ |

### 3.3 Deployment

| Component | Requirement | Implementation |
|-----------|-------------|----------------|
| Backend Host | Cloud VM | AWS EC2 ✅ |
| Frontend Host | CDN | Vercel ✅ |
| SSL Certificate | Required | ✅ |
| CI/CD | Automated | GitHub Actions ✅ |

---

## 4. Data Requirements

### 4.1 Data Sources

| Source | Type | Frequency |
|--------|------|-----------|
| Citi Bike trips | CSV | Monthly |
| Weather data | API | On-demand |
| Station locations | Static | As needed |

### 4.2 Data Volume

| Metric | Value |
|--------|-------|
| Total trips | ~68 million |
| Stations | 200+ |
| Time range | 2020-2025 |
| Features | 54 |

---

## 5. Acceptance Criteria

### 5.1 Model Performance

| Criterion | Threshold | Achieved |
|-----------|-----------|----------|
| R² Score | > 0.70 | ✅ 0.781 |
| MAE | < 5.0 | ✅ 2.66 |
| Training Time | < 5 min | ✅ ~2 min |

### 5.2 System Performance

| Criterion | Threshold | Achieved |
|-----------|-----------|----------|
| API Endpoints | All working | ✅ 17/17 |
| Error Rate | < 1% | ✅ |
| Response Time | < 3s | ✅ |

---

## 6. Constraints

1. **Budget:** AWS Free Tier (t3.micro instance)
2. **Data:** Public data only (no proprietary sources)
3. **Time:** Project completed within semester timeline
4. **Team Size:** Individual project
