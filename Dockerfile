# First stage: build dependencies
FROM python:3.8-slim-buster AS build

# Set the working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ gfortran libffi-dev libssl-dev libopenblas-dev

# Copy the requirements file into the container
COPY requirements.txt .

# Install Python dependencies
RUN pip install --user --no-cache-dir -r requirements.txt

# Second stage: create the final image
FROM python:3.8-slim-buster

# Copy installed Python packages from the first stage
COPY --from=build /root/.local /root/.local

# Make sure scripts in .local are usable:
ENV PATH=/root/.local/bin:$PATH

# Set the working directory
WORKDIR /app

# Copy the rest of the application code into the container
COPY . .

# Expose the port your application will run on (if needed)
EXPOSE 8000

# Start the Jupyter Notebook server
CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8000", "--no-browser", "--allow-root"]
