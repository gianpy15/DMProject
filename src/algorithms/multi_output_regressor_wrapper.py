from sklearn.multioutput import MultiOutputRegressor

class MultiOutputRegressorWrapper():

    def __init__(self, model, X, y):

        self.model = model
        self.X = X
        self.y = y
        self.multioutputregressor = None

    def fit(self):
        print('starting the fit')
        self.multioutputregressor = MultiOutputRegressor(self.model, n_jobs=-1).fit(self.X, self.y)

    def predict_train(self):
        pred = self.multioutputregressor.predict(X=self.X)
        return pred

