from sklearn import linear_model
from src.algorithms.multi_output_regressor_wrapper import MultiOutputRegressorWrapper
from src.algorithms.multi_output_regressor_wrapper import MultiOutputRegressionChainWrapper
import src.data as data
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import pandas as pd
import numpy as np


class LassoRegression():
    def __init__(self, alpha):
        self.alpha = alpha
        self.model = linear_model.Lasso(alpha=self.alpha, max_iter=10000)

    def get_params(self, deep):
        return {'alpha':self.alpha}

    def set_params(self):
        pass

    def fit(self, X, y):
        print('fitting lasso model')
        X = X[y > 0]
        y = y[y > 0]
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, shuffle=False)
        self.model.fit(X_train, y_train)
        y_hat = self.model.predict(X_val)
        print(mean_absolute_error(y_val, y_hat))


    def predict(self, X):
        return self.model.predict(X)

if __name__ == '__main__':
    X, y = data.dataset('train')
    X = X.fillna(0)
    y = y.fillna(0)
    base_model = LassoRegression(alpha=2)
    model_wrapper = MultiOutputRegressorWrapper(base_model, X, y)
    model_wrapper.fit()
