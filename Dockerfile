FROM python:3.11-slim
 
WORKDIR /app
 
# install dependencies first (cached layer)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
 
# pre-download the embedding model during build
# avoids cold-start delay and network dependency at runtime
RUN python -c "from fastembed import TextEmbedding; list(TextEmbedding('sentence-transformers/all-MiniLM-L6-v2').embed(['warmup']))"
 
# copy app code
COPY . .
 
EXPOSE 8000
 
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]