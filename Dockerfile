# First stage: build dependencies
FROM python:3.8 AS build

# Set the working directory
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install Python dependencies
RUN pip install --user --no-cache-dir -r requirements.txt

# Second stage: create the final image
FROM python:3.8

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

# Start the application
CMD ["python3", "Main.py"]
