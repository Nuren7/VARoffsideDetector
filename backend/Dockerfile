# Använd en lättviktsbaserad Python 3.10-bild
FROM python:3.10-slim

# Sätt arbetskatalog i containern
WORKDIR /app

# Installera systemberoenden som krävs för OpenCV (Computer Vision)
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Kopiera och installera Python-beroenden från requirements.txt
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Kopiera över resten av källkoden till containern
COPY . .

# Exponera porten som FastAPI använder
EXPOSE 8000

# Starta FastAPI-servern med Uvicorn
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]