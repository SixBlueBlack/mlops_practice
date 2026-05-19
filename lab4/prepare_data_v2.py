import pandas as pd

data = pd.read_csv('titanic.csv')
mean_age = data['Age'].mean()
data['Age'] = data['Age'].fillna(mean_age)

data.to_csv('titanic.csv', index=False)
print("Версия 2: пропуски в Age заполнены средним.")