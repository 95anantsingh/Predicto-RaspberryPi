

import socket
import time
import pandas as pd
import multiprocessing
from threading import Thread
from sklearn import metrics
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import train_test_split

IP = "192.10.9.96"
controlPort = 2210

rxCommandData = 0
txResultData = 0

def diabetes():
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


def breastCancer():
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
    print metrics.accuracy_score(expected, prediction)

    textureMean = 11.37
    perimeterMean = 134.75
    smoothnessMean = 0.1234
    compactness = 0.2145
    symmetryMean = 0.252

    sample_tumor = [[textureMean, perimeterMean, smoothnessMean, compactness, symmetryMean]]

    predictor = model.predict(sample_tumor)

    print "Prediction: "
    print(predictor)


def dataUpdater():
    global IP, controlPort, rxCommandData
    print '\ndataUpdater Started'
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((IP, controlPort))
    while True:
        data, address = sock.recvfrom(1024)
        dataString = data.decode('utf-8')
        rxCommandData = int(dataString)

        msg = str(txResultData)
        sendBytes = msg.encode('utf-8')
        sock.sendto(sendBytes, address)


def dataController():
    global txResultData
    print '\ndataController Started'
    rx = 100
    print rxCommandData
    print rx
    while True:
        if rx != rxCommandData:
            txResultData += 1
            print "CommandData: %d" % rxCommandData
            rx = rxCommandData


if __name__ == '__main__':
    print '\nStarting Processes...'
    dataUpdaterProcess = multiprocessing.Process(target=dataUpdater)
    dataControllerProcess = multiprocessing.Process(target=dataController)
    dataUpdaterProcess.start()
    dataControllerProcess.start()

    while True:
        a = int(input())
        if a == 1:
            dataUpdaterProcess.terminate()
            dataControllerProcess.terminate()