# Stage 1: Build stage
FROM python:3.12-slim as builder

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV HF_HOME=/app/cache
ENV PATH="/root/.local/bin:$PATH"

# Install build dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Stage 2: Runtime stage
FROM python:3.12-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV HF_HOME=/app/cache
ENV PATH="/root/.local/bin:$PATH"

# Set working directory
WORKDIR /app

# Copy only necessary files from builder
COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application code and model
COPY app.py .
COPY fine_tuned_model fine_tuned_model

# Create cache directory
RUN mkdir -p /app/cache && chmod 777 /app/cache

# Expose port
EXPOSE 8000

# Set default command
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
