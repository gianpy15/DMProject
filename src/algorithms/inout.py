import joblib
import numpy as np
from sklearn.metrics import mean_absolute_error

def save(model, path):
    joblib.dump(model, path)

def load(path):
    return joblib.load(path)

def evaluate(model, X_test, y_test, intermediate=False):
    print('This evaluation will be done for samples having ALL y not null!')
    # mask_test = np.all(y_test.notnull(), axis=1)
    # y_pred = model.predict(X_test[mask_test])
    y_pred = model.predict(X_test)

    MAE = []
    for i in range(y_test.shape[1]):
        # evaluate on only non null targets per column
        mask_not_null = y_test.iloc[:,i].notnull()

        MAE.append( mean_absolute_error(y_test[mask_not_null].iloc[:,i], y_pred[mask_not_null,i]) )
    if intermediate:
        return np.mean(MAE), MAE
    else:
        return np.mean(MAE)
