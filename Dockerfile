FROM python:3.12-slim

WORKDIR /app

# Dependencias del sistema
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ curl && \
    rm -rf /var/lib/apt/lists/*

# Instalar dependencias Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar código fuente
COPY *.py ./
COPY entrypoint.sh .
RUN chmod +x entrypoint.sh

# Crear directorios de datos
RUN mkdir -p models outputs/plots outputs/reports outputs/signals outputs/history

# Variables de entorno configurables
ENV TRAIN_INTERVAL_HOURS=4
ENV API_PORT=8000

EXPOSE 8000

# Healthcheck
HEALTHCHECK --interval=60s --timeout=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

ENTRYPOINT ["bash", "entrypoint.sh"]
