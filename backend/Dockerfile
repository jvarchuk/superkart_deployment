FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

# Install dependencies first so Docker can cache this layer
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the API code and the serialized model
COPY . .

EXPOSE 7860

# Gunicorn serves the Flask app object `superkart_api` defined in app.py
CMD ["gunicorn", "-w", "2", "--timeout", "60", "-b", "0.0.0.0:7860", "app:superkart_api"]
