FROM ghcr.io/astral-sh/uv:0.7.2-python3.12-bookworm-slim

WORKDIR /app

COPY . /app/

RUN uv pip install --system --no-cache-dir -r requirements.txt 


EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]