
# data analysis and wrangling
import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split
# For evaluating our ML results
from sklearn import metrics
from sklearn.naive_bayes import GaussianNB

d = pd.read_csv('Diabetes_data.csv')

X = d.drop(['Outcome'], axis=1)
Y = d.Outcome

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=.44)

model = GaussianNB()
model.fit(X_train, Y_train)

predicted = model.predict(X_test)
expected = Y_test

print "Accuracy: "
print metrics.accuracy_score(expected, predicted)

pregnancies = 1
glucose = 85
bloodPressure = 66
skinThickness = 29
insulin = 0
bmi = 26
diabetesPedigreeFunction = 0.351
age = 31
sample_woman = [[pregnancies, glucose, bloodPressure, skinThickness, insulin, bmi, diabetesPedigreeFunction, age]]

prediction = model.predict(sample_woman)

print "Prediction: "
print prediction
