FROM python:3.11-slim

LABEL description="Микросервис для анализа эмоций на основе BERT"

WORKDIR /app

COPY requirements.txt .

RUN pip install --upgrade pip \
    && pip install -r requirements.txt

COPY app.py .
COPY test_predict.py .

EXPOSE 1460

CMD ["streamlit", "run", "app.py", "--server.port", "1460", "--server.address", "0.0.0.0"]
