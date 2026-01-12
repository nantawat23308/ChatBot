# --- Stage 1: Build dependencies ---
FROM python:3.12-slim-bookworm as builder

# Set environment variables to prevent Python from writing .pyc files & buffering logs
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install system dependencies needed for building some Python packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install python dependencies into a local folder
COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt



FROM python:3.12-slim-bookworm

WORKDIR /app

# Copy only the installed dependencies from the builder stage
COPY --from=builder /install /usr/local

# Create a non-root user for security (Production Best Practice)
RUN addgroup --system appuser && adduser --system --group appuser

# Copy application code
# --- FIXING THE COPY LOGIC ---
# Since main.py is in your project root:
COPY main.py .
# Since the 'app' folder is in your project root:
COPY . .


# Change ownership to the non-root user
RUN chown -R appuser:appuser /app
USER appuser

# Expose the port FastAPI will run on
EXPOSE 8000

# Use Gunicorn with Uvicorn workers for production process management
CMD ["gunicorn", "-k", "uvicorn.workers.UvicornWorker", "--bind", "0.0.0.0:8000", "main:app", "--workers", "4"]