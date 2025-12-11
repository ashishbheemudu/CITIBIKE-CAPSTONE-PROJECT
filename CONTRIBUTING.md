# Contributing to NYC Citi Bike Analytics Dashboard

Thank you for considering contributing to this project! This document provides guidelines for contributing.

## Table of Contents
- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Submitting Changes](#submitting-changes)
- [Coding Standards](#coding-standards)
- [Testing](#testing)

## Code of Conduct

This project adheres to a code of conduct. By participating, you are expected to uphold this code.

## Getting Started

1. Fork the repository
2. Clone your fork: `git clone https://github.com/YOUR_USERNAME/CITIBIKE-CAPSTONE-PROJECT.git`
3. Create a branch: `git checkout -b feature/your-feature-name`

## Development Setup

### Backend
```bash
cd backend
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
uvicorn main:app --reload
```

### Frontend
```bash
cd frontend
npm install
npm run dev
```

## Submitting Changes

1. Ensure all tests pass: `pytest` (backend) and `npm test` (frontend)
2. Update documentation if needed
3. Commit with clear messages following conventional commits:
   - `feat:` for new features
   - `fix:` for bug fixes
   - `docs:` for documentation
   - `test:` for tests
   - `refactor:` for refactoring
4. Push to your fork and submit a Pull Request

## Coding Standards

### Python (Backend)
- Follow PEP 8
- Use type hints for all functions
- Add docstrings to all functions and classes
- Maximum line length: 120 characters

### JavaScript (Frontend)
- Use ES6+ syntax
- Add PropTypes to all components
- Add JSDoc comments to functions
- Use meaningful variable names

## Testing

### Backend
```bash
cd backend
pytest -v
pytest --cov=. --cov-report=html
```

### Frontend
```bash
cd frontend
npm test
npm run lint
```

## Questions?

Open an issue or contact the maintainers.
