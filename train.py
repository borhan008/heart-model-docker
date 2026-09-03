

import pandas as pd
import numpy as np

df = pd.read_csv("/content/heart.csv");
df.head()

import sklearn as sk
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

X = df.drop('target', axis=1)
Y = df['target']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

model = RandomForestClassifier()
model.fit(X_train, Y_train)

test = model.predict(X_test)
accuracy_score(test, Y_test)

import joblib
joblib.dump(model, 'model.joblib')