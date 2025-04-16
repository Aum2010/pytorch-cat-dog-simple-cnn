# Use the official Python image from the Docker Hub
# FROM python:3.10-slim
FROM python:3.13.3

# Set the working directory
WORKDIR /workspace

# Install system dependencies for OpenCV
RUN apt-get update && apt-get install -y libgl1-mesa-glx libglib2.0-0 && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install torch torchvision torchaudio matplotlib pandas numpy
# RUN pip install ultralytics opencv-python
