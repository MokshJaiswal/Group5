# Base Python image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy entire app folder
COPY app/ ./app/

# Set working directory to app/
WORKDIR /app/app

# Expose Flask port
EXPOSE 8000

# Run Flask app
CMD ["python", "main.py"]
