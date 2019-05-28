
import socket
import pandas as pd
from threading import Thread
from sklearn import metrics
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

IP = "192.10.9.96"
controlPort = 2410

rxCommandData = ""
BrCnModel = RandomForestClassifier(n_estimators=100)
DiFeModel = GaussianNB()
DiMaModel = GaussianNB()
BrCnAccuracy = 0
DiFeAccuracy = 0
DiMaAccuracy = 0


def diabetesF():
    global DiFeModel, DiFeAccuracy
    d = pd.read_csv('Diabetes_female_data.csv')

    X = d.drop(['Outcome'], axis=1)
    Y = d.Outcome

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=.44)

    model = GaussianNB()
    model.fit(X_train, Y_train)

    predicted = model.predict(X_test)
    expected = Y_test

    DiFeAccuracy = 100 * (metrics.accuracy_score(expected, predicted))
    DiFeModel = model


def diabetesM():
    global DiMaModel, DiMaAccuracy
    d = pd.read_csv('Diabetes_male_data.csv')

    X = d.drop(['Outcome'], axis=1)
    Y = d.Outcome

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=.44)

    model = GaussianNB()
    model.fit(X_train, Y_train)

    predicted = model.predict(X_test)
    expected = Y_test

    DiMaAccuracy = 100 * (metrics.accuracy_score(expected, predicted))
    DiMaModel = model


def breastCancer():
    global BrCnAccuracy, BrCnModel
    data = pd.read_csv("Breast_Cancer_data.csv", header=0)
    data['diagnosis'] = data['diagnosis'].map({'M': 1, 'B': 0})

    X = data.drop(['diagnosis'], axis=1)
    Y = data.diagnosis

    train_X, test_X, train_y, test_y = train_test_split(X, Y, test_size=0.3)

    model = RandomForestClassifier(n_estimators=100)
    model.fit(train_X, train_y)

    prediction = model.predict(test_X)
    expected = test_y

    BrCnAccuracy = 100 * (metrics.accuracy_score(expected, prediction))
    BrCnModel = model


def dataUpdater():
    global IP, controlPort, rxCommandData
    print('\ndataUpdater Started')
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((IP, controlPort))
    while True:
        data, address = sock.recvfrom(1024)
        dataString = data.decode('utf-8')
        rxCommandData = dataString
        print("Receive: " + rxCommandData)

        msg = dataController()
        sendBytes = msg.encode('utf-8')
        sock.sendto(sendBytes, address)
        print("Send: " + msg)


def dataController():
    functionNumber = findData("[", "]")
    if functionNumber == 1:
        tumorData = ([findData("a", "b"), findData("b", "c"), findData("c", "d"), findData("d", "e"),
                      findData("e", "f"), ])
        txResultData = BrCnModel.predict(tumorData)
        txAccuracy = BrCnAccuracy
    elif functionNumber == 2:
        diabetesData = ([findData("a", "b"), findData("b", "c"), findData("c", "d"), findData("d", "e"),
                         findData("e", "f"), findData("f", "g"), findData("g", "h"), findData("h", "i")])
        txResultData = DiFeModel.predict(diabetesData)
        txAccuracy = DiFeAccuracy
    elif functionNumber == 3:
        diabetesData = ([findData("a", "b"), findData("b", "c"), findData("c", "d"), findData("d", "e"),
                         findData("e", "f"), findData("f", "g"), findData("g", "h")])
        txResultData = DiMaModel.predict(diabetesData)
        txAccuracy = DiMaAccuracy
    else:
        txResultData = 0
        txAccuracy = 0
    return str(txResultData) + str(txAccuracy)


def findData(s1, s2):
    start = rxCommandData.find(s1)+1
    end = rxCommandData.find(s2)
    return float(rxCommandData[start:end])


if __name__ == '__main__':
    print('\nStarting Threads...')
    breastCancer()
    diabetesF()
    diabetesM()
    dataUpdaterThread = Thread(target=dataUpdater)
    dataUpdaterThread.start()
