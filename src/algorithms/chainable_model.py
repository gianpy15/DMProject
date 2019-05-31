import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

class ChainableModel():
    """
        Base class for the models that can be used in a multioutput or chain model.
    """

    def __init__(self, params_dict):
        """ Init the model. Params_dict should contain the parameters that will be passed
            to the init of the underline model. In addition to those, some other params can be
            specified:
            
            X (DataFrame):      train dataframe
            name (string):      model name
            val_split (float):  validation size (default: 0.2)
        """
        assert 'X' in params_dict, 'X param is compulsory!'

        self.params_dict = params_dict

        # workaround
        X = params_dict.pop('X')
        self.columns = X.columns
        self.col_dtypes = { col:str(X[col].dtype) for col in X.columns }
        
        self.name = params_dict.pop('name', 0)
        self.val_split = params_dict.pop('val_split', 0.2)
        
        # initialize the model with the correct dictionary
        self.model = self.build_model(self.params_dict)

        # reset the dictionary including X
        self.params_dict['X'] = X
        
        self.eval_res = {}
    
    def build_model(self, params_dict):
        """ Override to build the custom model here.
            Return a model.
        """
        #Ex: return CatBoostRegressor(**params_dict)
        return object()

    def get_params(self, deep=True):
        return {'params_dict': self.params_dict }

    def set_params(self):
        pass

    def fit(self, X, y):
        X = pd.DataFrame(X, columns=self.columns).astype(self.col_dtypes)
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=self.val_split, shuffle=False)

        # filters NaN targets in train and validation
        mask = np.isfinite(y_train.astype('float'))
        x = X_train[mask]
        y = y_train[mask]

        mask_val = np.isfinite(y_val.astype('float'))
        x_val = X_val[mask_val]
        y_val = y_val[mask_val]
        
        print('{}: fitting...'.format(self.name))
        self.fit_model(x, y, x_val, y_val)

    def fit_model(self, X, y, X_val, y_val):
        """
            Override to implement a custom model fit function.
        """
        pass

    def validate(self):
        pass

    def predict(self, X):
        return self.model.predict(X)
