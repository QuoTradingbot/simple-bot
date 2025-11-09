# Use Python 3.11 slim image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy requirements
COPY cloud-api/requirements-signal.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements-signal.txt

# Copy application files
COPY cloud-api/signal_engine_v2.py .
COPY cloud-api/database.py .
COPY cloud-api/redis_manager.py .

# Copy admin dashboard
COPY admin-dashboard/ ./admin-dashboard/

# Copy Kevin's proven RL experiences (6,880 signals + 2,961 exits)
COPY data/signal_experience.json .
COPY data/exit_experience.json .

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
  CMD python -c "import requests; requests.get('http://localhost:8000/health')"

# Run the signal engine
CMD ["uvicorn", "signal_engine_v2:app", "--host", "0.0.0.0", "--port", "8000"]
