import os
import pandas as pd
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

os.makedirs('train', exist_ok=True)
os.makedirs('test', exist_ok=True)

digits = load_digits()
df = pd.DataFrame(digits.data, columns=[f'pixel{i}' for i in range(64)])
df.insert(0, 'label', digits.target)

train, test = train_test_split(df, test_size=0.2, random_state=22)

train.to_csv('train/digits_train.csv', index=False)
test.to_csv('test/digits_test.csv', index=False)

print("Обучающий и тестовый наборы сохранены.")
