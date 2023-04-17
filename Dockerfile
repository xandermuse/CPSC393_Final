# Use the official Python base image
FROM python:3.8

# Set the working directory
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install the dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code into the container
COPY . .

# Expose the port your application will run on (if needed)
EXPOSE 8000

# Start the application
CMD ["python", "Main.py"]
