from catboost.datasets import titanic

train_df, _ = titanic()

data = train_df[['Pclass', 'Sex', 'Age', 'Survived']].copy()

data.to_csv('titanic.csv', index=False)
print("Версия 1: исходные данные сохранены.")