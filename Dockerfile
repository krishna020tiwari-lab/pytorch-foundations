FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

RUN pip install --no-cache-dir fastapi uvicorn onnxruntime numpy pydantic

COPY . /app
EXPOSE 8000

CMD ["uvicorn", "Pytorch_Foundation.16_fastapi_onnx_service:app", "--host", "0.0.0.0", "--port", "8000"]
