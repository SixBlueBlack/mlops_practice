import pandas as pd

data = pd.read_csv('titanic.csv')

data = pd.get_dummies(data, columns=['Sex'], prefix='Sex')

data.to_csv('titanic.csv', index=False)
print("Версия 3: добавлены признаки Sex_female, Sex_male.")