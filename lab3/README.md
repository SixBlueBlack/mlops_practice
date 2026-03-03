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

### Тестирование

python test_predict.py

### Требования
Python 3.11+
Docker 20.10+
Docker Compose 2.0+


## Проверка работы

После создания всех файлов выполните:

```bash
# Перейдите в папку lab3
cd lab3

# Соберите и запустите контейнер
docker-compose up --build

# В другом терминале проверьте тесты
docker-compose exec emotion-analysis python test_predict.py

Откройте браузер и перейдите на http://localhost:1460 - вы должны увидеть интерфейс Streamlit.
