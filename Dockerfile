# Use a lightweight Python base image
FROM python:3.10-slim

# Set the working directory inside the container
WORKDIR /RFS-app

# Copy the deployment requirements first to leverage Docker cache
COPY requirements_deployment.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements_deployment.txt

# Copy the application code and required directories
COPY app/ ./app/


# Copy only the final fine-tuned model to keep the image size smaller
COPY models/flan_t5_xray_final/ ./models/flan_t5_xray_final/

# Expose port 8080 (Required for Google Cloud Run)
EXPOSE 8080

# Command to run the API
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]

# Image metadata
LABEL name="RFS-app"