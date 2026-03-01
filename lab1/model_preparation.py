import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import pickle

# Загрузка предобработанных данных
train_preprocessed = pd.read_csv("train/preprocessed/digits_train_preprocessed.csv")

X_train = train_preprocessed.drop(columns=['label'])
y_train = train_preprocessed['label']

test_preprocessed = pd.read_csv("test/preprocessed/digits_test_preprocessed.csv")
X_test = test_preprocessed.drop(columns=['label'])
y_test = test_preprocessed['label']

model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)

train_accuracy = model.score(X_train, y_train)
test_accuracy = model.score(X_test, y_test)

# Сохранение модели
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)
