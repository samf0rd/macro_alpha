# Use official lightweight Python image
FROM python:3.12-slim

# Set the working directory in the container
WORKDIR /app

# Install system dependencies (required for some ML packages)
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*
    # build-essential installs the required compilers (gcc, make, etc.) so those ML packages can build themselves
    # "Once you install the tools, throw the installation wrappers in the trash." This keeps your Docker image small, which saves you money on AWS.

# Copy requirements first (to leverage Docker cache)
COPY requirements.txt .

# Set environment variables to prevent SHAP from installing PyTorch
ENV SHAP_INSTALL_TORCH=0
ENV SHAP_INSTALL_LIGHTGBM=0

# Install Python dependencies
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
# Copy the rest of the project files
COPY . .

# Expose the port Streamlit uses
EXPOSE 8501

# Command to run the Streamlit app
CMD ["streamlit", "run", "dashboard/app.py", "--server.port=8501", "--server.address=0.0.0.0"]