import pandas as pd
from sklearn.preprocessing import StandardScaler
import os
import joblib

os.makedirs('train/preprocessed', exist_ok=True)
os.makedirs('test/preprocessed', exist_ok=True)

# Загрузка обучающих данных
train_data = pd.read_csv("train/digits_train.csv")

# Загрузка тестовых данных
test_data = pd.read_csv("test/digits_test.csv")

# Выделение признаков и целевой переменной для обучающей выборки
X_train = train_data.drop(columns=['label'])
y_train = train_data['label']

# Выделение признаков и целевой переменной для тестовой выборки
X_test = test_data.drop(columns=['label'])
y_test = test_data['label']

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

X_test_scaled = scaler.transform(X_test)

# Обучающая выборка
train_preprocessed = pd.DataFrame(X_train_scaled, columns=X_train.columns)
train_preprocessed.insert(0, 'label', y_train.values)
train_preprocessed.to_csv('train/preprocessed/digits_train_preprocessed.csv', index=False)

# Тестовая выборка
test_preprocessed = pd.DataFrame(X_test_scaled, columns=X_test.columns)
test_preprocessed.insert(0, 'label', y_test.values)
test_preprocessed.to_csv('test/preprocessed/digits_test_preprocessed.csv', index=False)

joblib.dump(scaler, 'scaler.pkl')
