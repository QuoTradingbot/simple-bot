# Use Python 3.11 slim image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy requirements
COPY requirements-azure.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements-azure.txt

# Copy application files
COPY main.py .
COPY signal_engine.py .
COPY database.py .
COPY models.py .

# Expose port
EXPOSE 8000

# Run the application
CMD ["gunicorn", "main:app", "--workers", "2", "--worker-class", "uvicorn.workers.UvicornWorker", "--bind", "0.0.0.0:8000"]
