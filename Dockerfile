FROM python:3.11-slim

WORKDIR /app

RUN useradd -m appuser

COPY app/ /app/

RUN pip install --no-cache-dir fastapi==0.115.6 uvicorn[standard]==0.30.6 httpx==0.27.2 jinja2==3.1.4

USER appuser

EXPOSE 55556

CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${APP_PORT:-55556}"]
