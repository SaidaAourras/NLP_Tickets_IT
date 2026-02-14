FROM python:3.10-slim

WORKDIR /app

# Packages système
RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential && \
    rm -rf /var/lib/apt/lists/*

# Installation Python packages
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
    numpy pandas scikit-learn nltk sentence-transformers joblib

# NLTK data
RUN python -c "import nltk; nltk.download('stopwords'); nltk.download('punkt')"

# Code source
COPY src/ ./src/

# Répertoires
RUN mkdir -p /app/data/raw /app/data/processed /app/outputs /app/models /app/chromadb

# Modifier le chemin par défaut pour pointer vers data/raw/
CMD ["python", "src/pipeline.py", "/app/data/raw/dataset.csv"]