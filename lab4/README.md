# Версионирование наборов данных

## S3

Было выбрано MinIO S3-хранилище, потому что мне Google Disk запрещает выполнять dvc push, нужен сервисный аккаунт, а для него нужна зарубежная карточка, которой у меня нет

## Порядок действий

```bash

pip install dvc

dvc init

# Первый коммит в Git
git add .
git commit -m "Init DVC project"

# Создание prepare_data_v1.py

dvc add lab4/titanic.csv

git add lab4/prepare_data_v1.py lab4/titanic.csv.dvc lab4/.gitignore
git commit -m "V1"

# Запуск и настройка MinIO
docker run -p 9000:9000 -p 9001:9001 --name minio -e "MINIO_ROOT_USER=minioadmin" -e "MINIO_ROOT_PASSWORD=minioadmin" quay.io/minio/minio server /data --console-address ":9001"

pip install dvc-s3

dvc remote add -d myremote s3://dvc-store

dvc remote modify myremote endpointurl http://127.0.0.1:9000

dvc remote modify myremote access_key_id minioadmin
dvc remote modify myremote secret_access_key minioadmin

dvc remote modify myremote use_ssl false

git add .dvc/config
# Тут Switch, потому что изначально был гугл диск, который не заработал
git commit -m "Switch to local MinIO S3 remote"

# Теперь push работает
dvc push

# Для prepare_data_v2.py и prepare_data_v3.py действуем аналогично

# Для переключения между версиями датасета выполняем команды:
git checkout хеш коммита
dvc checkout