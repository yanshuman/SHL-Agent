FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir 'huggingface-hub<0.24'

COPY data/ data/
COPY src/ src/

# Pre-build the FAISS index at build time so cold start is fast
RUN python -c "from src.retriever import CatalogRetriever; CatalogRetriever('data/catalog.json', 'data/catalog.index')"

ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app/src

EXPOSE 8000

CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]
