#!/bin/bash
set -e

echo "================================"
echo "🚀 Starting ML Service"
echo "================================"
echo "PORT: ${PORT:-5000}"
echo "RAILWAY_ENVIRONMENT: ${RAILWAY_ENVIRONMENT:-local}"
echo "Python version: $(python --version)"
echo "================================"

# Start the application using uvicorn CLI (Railway recommended way)
exec uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-5000}
