FROM python:3.11-slim

WORKDIR /app

# Install system deps (if needed) and pip dependencies
COPY requirements.txt ./
RUN python -m pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy project
COPY . .

ENV PYTHONUNBUFFERED=1

EXPOSE 8000 8501

# Default command runs the API; docker-compose will override for streamlit
CMD ["uvicorn", "src.serve.api:app", "--host", "0.0.0.0", "--port", "8000"]
