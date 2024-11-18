# Use a lightweight Python image
FROM python:3.13

# Set working directory
ADD app.py .
WORKDIR /app

COPY libraries.txt .
# Install dependencies

RUN pip3 install -r libraries.txt

COPY . .

# Start the Flask application on the required port
CMD ["python3", "app.py"]