# syntax=docker/dockerfile:1.2
FROM python:3.9-slim

# Update system dependencies
RUN apt-get update && \
    apt-get install -y build-essential gcc libssl-dev

# Set the working directory
WORKDIR /delay_api

# Copy the requirements file
COPY requirements.txt .

# Install dependencies
RUN pip install -r requirements.txt

# Copy all .py files from the challenge folder
# This will copy `model.py`, `api.py`, and `__init__.py`
COPY challenge/*.py /delay_api/challenge/

# Copy model file
COPY models/model.json /delay_api/models/model.json

# Expose port 8080
EXPOSE 8080

# Run FastAPI application with Uvicorn
CMD ["uvicorn", "challenge.api:app", "--host", "0.0.0.0", "--port", "8080"]
