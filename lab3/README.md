# Анализ эмоций в тексте

Микросервис для анализа эмоций в русскоязычном тексте на основе BERT модели.

## Возможности

- Определение 7 эмоций: neutral, happiness, sadness, enthusiasm, fear, anger, disgust
- Веб-интерфейс на Streamlit
- REST API для интеграции

## Быстрый старт

### Запуск с помощью Docker

```bash
# Сборка образа
docker build -t emotion-analysis .

# Запуск контейнера
docker run -p 1460:1460 emotion-analysis
