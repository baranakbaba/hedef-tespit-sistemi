# Hedef Tespit ve Takip Sistemi - Docker imaji
# Kullanim:
#   docker build -t hedef-tespit-sistemi .
#   docker run -p 8501:8501 hedef-tespit-sistemi

FROM python:3.11-slim

# OpenCV'nin ihtiyac duydugu sistem kutuphaneleri
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
