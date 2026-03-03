import os
import pandas as pd
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
import requests
import io

os.makedirs('train', exist_ok=True)
os.makedirs('test', exist_ok=True)

# Альтернатива 1: загрузка из sklearn
digits = load_digits()
df = pd.DataFrame(digits.data, columns=[f'pixel{i}' for i in range(64)])
df.insert(0, 'label', digits.target)

# Альтернатива 2: загрузка из GitHub (на случай если sklearn не установлен)
# url = 'https://raw.githubusercontent.com/scikit-learn/scikit-learn/main/sklearn/datasets/data/digits.csv.gz'
# response = requests.get(url)
# data = pd.read_csv(io.BytesIO(response.content), header=None)

train, test = train_test_split(df, test_size=0.2, random_state=22)
train.to_csv('train/digits_train.csv', index=False)
test.to_csv('test/digits_test.csv', index=False)

print("Данные успешно загружены и сохранены")
