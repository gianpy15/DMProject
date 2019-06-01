import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

class ChainableModel():
    """
        Base class for models that can be used in a multioutput or chain model.
    """

    def __init__(self, params_dict):
        """ Init the model. Params_dict should contain the parameters that will be passed
            to the init of the underline model. In addition to those, some other params can be
            specified:
            
            X (DataFrame):      train dataframe
            name (string):      model name
            val_split (float):  validation size (default: 0.2)
        """
        # X is needed to be passed as an addtional param in order to 
        # save the dataframe columns and types
        assert 'X' in params_dict, 'X param is compulsory!'

        self.params_dict = params_dict

        # in order to not lose the original dataframe structure when the data are
        # passed to the underlying models, store the columns names and dtypes
        X = params_dict.pop('X')
        self.columns = list(X.columns)
        self.col_dtypes = { col:str(X[col].dtype) for col in X.columns }
        
        self.name = params_dict.pop('name', 0)
        self.val_split = params_dict.pop('val_split', 0.2)
        
        # initialize the model using the dictionary without the additional params
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
        # when using the regressor chain, the number of columns is increased by one
        # at each step of the chain, so the columns variable should be adapted accordingly
        columns = self.columns
        order = X.shape[1] - len(columns)
        if order > 0:
            # add additional "dummy" columns to the dataframe
            prev_pred_columns = ['$step_{}$'.format(i) for i in range(order)]
            columns.extend(prev_pred_columns)

        # rebuild the original dataframe with the real dtypes for each column
        X = pd.DataFrame(X, columns=columns).astype(self.col_dtypes)
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
