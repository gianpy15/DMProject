from sklearn.multioutput import MultiOutputRegressor
from sklearn.multioutput import RegressorChain

class MultiOutputRegressorWrapper():

    def __init__(self, model, X, y):

        self.model = model
        self.X = X
        self.y = y
        self.multioutputregressor = MultiOutputRegressor(self.model)

    def fit(self):
        print('starting the fit')
        self.multioutputregressor.fit(self.X, self.y)

    def predict(self, X):
        pred = self.multioutputregressor.predict(X=X)
        return pred

class MultiOutputRegressionChainWrapper():
    def __init__(self, model, X, y):
        self.model = model
        self.X = X
        self.y = y
        self.multioutputregressor = RegressorChain(self.model, cv=2)

    def fit(self):
        print('starting the fit')
        self.multioutputregressor.fit(self.X, self.y)
