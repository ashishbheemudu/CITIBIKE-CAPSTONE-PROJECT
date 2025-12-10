
# Use lightweight Python 3.9 image
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies (for building numpy/pandas if needed)
# Install system dependencies
# Removed build-essential to save space (Wheels available)
# libgomp1 is required for XGBoost/LightGBM OpenMP support
RUN apt-get update && apt-get install -y \
    curl \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy dependencies first for caching
COPY requirements.txt .

# Install Python dependencies
# Use --no-cache-dir to keep image small
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose port (Render uses 10000 by default but we map to 8000)
EXPOSE 8000

# Run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
