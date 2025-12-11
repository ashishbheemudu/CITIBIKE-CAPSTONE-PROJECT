# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [3.0.0] - 2025-12-11

### Added
- Modern lifespan context manager replacing deprecated `@app.on_event`
- Pydantic input validation models for API security
- CORS whitelist with specific origins (security improvement)
- Comprehensive README with architecture diagram
- GitHub Actions CI/CD pipeline
- Unit tests for prediction service
- API endpoint integration tests
- `.env.example` for environment variable documentation
- `.editorconfig` for consistent code formatting
- Pre-commit hooks configuration
- API documentation in docs/

### Changed
- Replaced `print()` statements with proper logging
- Switched to environment variables for API configuration
- Improved error handling across all endpoints

### Fixed
- Prediction.jsx undefined error variable bug
- Batch feature creation now enabled (5x faster)
- CORS security vulnerability (was allow_all)

### Removed
- Unused ThreadPoolExecutor
- Deprecated @app.on_event decorator
- Duplicate comments in frontend

## [2.0.0] - 2025-12-10

### Added
- Enhanced ML models (v3) with R² = 0.78
- XGBoost, LightGBM, CatBoost ensemble
- 54-feature prediction pipeline
- Advanced analytics endpoints (ABM, TFT, RL, etc.)

### Fixed
- Feature leakage in training (removed demand_change features)
- Model-feature count mismatch

## [1.0.0] - 2025-11-30

### Added
- Initial release
- FastAPI backend with 17 API endpoints
- React frontend with 8 pages
- Basic ML prediction model
- AWS EC2 deployment
- Vercel frontend deployment
