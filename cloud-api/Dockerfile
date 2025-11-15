# Use Python 3.11 slim image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy requirements
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY signal_engine_v2.py .
COPY database.py .
COPY redis_manager.py .
COPY neural_confidence.py .
COPY neural_exit.py .
COPY exit_params_config.py .

# Copy neural network models (your trained brains)
COPY neural_model.pth .
COPY exit_model.pth .

# Copy admin dashboard
COPY admin-dashboard/ ./admin-dashboard/

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
  CMD python -c "import requests; requests.get('http://localhost:8000/health')"

# Run the signal engine
CMD ["uvicorn", "signal_engine_v2:app", "--host", "0.0.0.0", "--port", "8000"]
