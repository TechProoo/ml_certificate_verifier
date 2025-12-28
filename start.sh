#!/bin/bash
set -e

echo "================================"
echo "🚀 Starting ML Service"
echo "================================"
echo "PORT: ${PORT:-5000}"
echo "RAILWAY_ENVIRONMENT: ${RAILWAY_ENVIRONMENT:-local}"
echo "Python version: $(python --version)"
echo "================================"

# Start the application
exec python -m app.main
