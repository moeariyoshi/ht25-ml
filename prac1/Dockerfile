# Use the official Python image
FROM python:3.9

# Set the working directory
WORKDIR /app

# Copy requirements first (for caching layers)
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application files
COPY . .

# Run app.py when the container launches
CMD ["python", "main.py"]
