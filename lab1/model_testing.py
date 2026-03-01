import pandas as pd
import pickle
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

test_data = pd.read_csv("test/preprocessed/digits_test_preprocessed.csv")

with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

X_test = test_data.drop(columns=['label'])
y_test = test_data['label']

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Model test accuracy is: {accuracy:.3f}")
