
import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics


print "start"
data = pd.read_csv("Breast_Cancer_data.csv", header=0)
data['diagnosis'] = data['diagnosis'].map({'M': 1, 'B': 0})

X = data.drop(['diagnosis'], axis=1)
Y = data.diagnosis

train_X, test_X, train_y, test_y = train_test_split(X, Y, test_size=0.3)

model = RandomForestClassifier(n_estimators=100)
model.fit(train_X, train_y)

prediction = model.predict(test_X)
expected = test_y

print "Accuracy: "
print metrics.accuracy_score(expected,prediction)
i = int(input("begin"))
textureMean = 11.37
perimeterMean = 134.75
smoothnessMean = 0.1234
compactness = 0.2145
symmetryMean = 0.252

sample_tumor = [[textureMean, perimeterMean, smoothnessMean, compactness,symmetryMean ]]

predictor = model.predict(sample_tumor)

print "Prediction: "
print(predictor)
