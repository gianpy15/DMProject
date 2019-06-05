import joblib
import numpy as np
from sklearn.metrics import mean_absolute_error

def save(model, path):
    joblib.dump(model, path)

def load(path):
    return joblib.load(path) 

def evaluate(model, X_test, y_test):
    print('This evaluation will be done for samples having ALL y not null!')
    mask_test = np.all(y_test.notnull(), axis=1)

    y_pred = model.predict(X_test[mask_test])
    return mean_absolute_error(y_test[mask_test], y_pred)