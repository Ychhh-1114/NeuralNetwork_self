from neuralnetwork import NeuralNetwork as nn
from activate_function import *
from loss import Loss
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn import metrics
from import_data import get_mnist

class NeuralNetwork(nn):
    def __init__(self):
        layerArr = []
        layerArr.append(nn.Layer(
            n_input = 28 * 28,
            n_output = 300,
            activator=ReLU(),
            learning_rate=0.01,
        ))
        layerArr.append(nn.Layer(
            n_input=300,
            n_output=100,
            activator=ReLU(),
            learning_rate=0.01,
        ))
        layerArr.append(nn.Layer(
            n_input=100,
            n_output=10,
            activator=ReLU(),
            learning_rate=0.01,
        ))
        super().__init__(
            layers=layerArr,
            loss=Loss.SoftmaxCrossEntroy(),
            EPOCH=30000
        )

    def predict(self,X):
        out = super().predict(X)
        return np.argmax(Loss.SoftmaxCrossEntroy().softmax(out),axis = 1).reshape(-1,1)



def processing_data(X):
    scaler = MinMaxScaler(feature_range=(0,1))
    return scaler.fit_transform(X)


X,y = get_mnist()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=0)
encoder = OneHotEncoder()
y_train = encoder.fit_transform(y_train)
X_train = processing_data(X_train)
X_test = processing_data(X_test)

print("running...")
model = NeuralNetwork()
model.fit(X_train,y_train)
pred_y = model.predict(X_test)
print("accuracy: ",metrics.accuracy_score(y_test,pred_y))

