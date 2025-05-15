import numpy as np


def linear(X, weights, bias):
    return np.dot(X, weights) + bias


class LinearRegression():

    def __init__(self, lr=0.0001, n_iters=1000):

        self.lr = lr
        self.n_iters = n_iters
        self.weights = None
        self.bias = None

    def fit(self, X, y):

        n_rows, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.n_iters):

            y_pred = linear(X, self.weights, self.bias)
            dw = (1 / n_rows) * np.dot(X.T, y_pred - y)
            db = (1 / n_rows) * np.sum(y_pred - y)

            self.weights -= self.lr * dw
            self.bias -= self.lr * db

    def predict(self, X):

        return linear(X, self.weights, self.bias)
