from sklearn.multioutput import MultiOutputRegressor

class MultiOutputRegressorWrapper():

    def __init__(self, model, X, y, X_val=None, y_val=None):

        self.model = model
        self.X = X
        self.y = y
        self.multioutputregressor = None
        self.X_val = X_val
        self.y_val = y_val

    def fit(self):
        self.multioutputregressor = MultiOutputRegressor(self.model)
        print('starting the fit')
        self.multioutputregressor.fit(self.X, self.y)

    def predict_train(self):
        pred = self.multioutputregressor.predict(X=self.X)
        return pred
