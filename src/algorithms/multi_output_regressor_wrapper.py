from sklearn.multioutput import MultiOutputRegressor
from src.algorithms.lightGBM import lightGBM
import src.data as data

class MultiOutputRegressorWrapper():

    def __init__(self, model):

        self.model = model

    def fit(self, X, y):
        print('starting the fit')
        multioutputregressor = MultiOutputRegressor(self.model).fit(X, y)

