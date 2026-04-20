# Use an official lightweight Python image
FROM python:3.10-slim

# Set the working directory
WORKDIR /app

# Copy requirements and install them
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy your ML model and the API script
COPY triage_model.pkl .
COPY api.py .

# Command to run the FastAPI app on port 8080 (required by Cloud Run)
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8080"]
