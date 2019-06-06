from abc import abstractmethod
from abc import ABC
from src.utils.folder import create_if_does_not_exist as check_folder
import pandas as pd
from src.utils.menu import yesno_choice
import os
from sklearn.preprocessing import MultiLabelBinarizer
import numpy as np

"""
extend this class and give an implementation to extract_feature to 
make available a new feature
"""
class FeatureBase(ABC):

    def __init__(self, mode, name, columns_to_onehot=[]):
        """
        columns_to_onehot: [(columns_header, onehot_mode), ...]
            onehot_mode: 'single' or 'multiple'
                'single': if we have just one categorical value for row
                'multiple': if we have multiple ones (assumes pipe separation)

        eg: [('action', 'single')]
        meaning that the header of the column to onehot is 'action' and the onehot modality is 'single'
        
        """
        self.cache = pd.DataFrame()
        self.mode = mode
        self.name = name
        self.columns_to_onehot = columns_to_onehot

    @abstractmethod
    def extract_feature(self):
        """
        Returns a dataframe that contains a feature (or more than one)
        on the first columns it should have an identifier of the single object to which the feature refers
        on the other column (or columns), the value of the features, with a meaningful name for the header.

        eg: road and key in the first two columns and the last columns for the avg speed in that sensor

        in case of categorical features, DO NOT RETURN A ONEHOT!
        In particular, return a single categorical value or a list of pipe-separated categorical values, and
        take care of setting self.columns_to_onehot nicely: base class will take care of one honetizing
        when read_feature is called.
        """
        pass

    def save_feature(self, overwrite_if_exists=None):
        """
        overwrite_if_exists: if true overwrite without asking; if false do not overwrite, if None ask before overwrite
        """
        path = 'resources/dataset/preprocessed/{}/features/{}/features.csv.gz'.format(self.mode, self.name)
        if os.path.exists(path):
            if overwrite_if_exists == None:
                choice = yesno_choice('The feature \'{}\' already exists. Want to recreate?'.format(self.name))
                if choice == 'n':
                    return
            elif not overwrite_if_exists:
                return
        df = self.extract_feature()
        check_folder(path)
        df.to_csv(path, index=False, compression='gzip')

    def post_loading(self, df):
        """ Callback called after loading of the dataframe from csv. Override to provide some custom processings. """
        return df

    def join_to(self, df, one_hot=False):
        """ Join this feature to the specified dataframe. The default implementation will join based on the
        common column between the 2 dataframes. Override to provide a custom join logic. """
        feature_df = self.read_feature(one_hot=one_hot)
        return pd.merge(df, feature_df)

    def read_feature(self, one_hot=False):
        """
        it reads a feature from disk and returns it.
        if one_hot = False, it returns it as was saved.
        if one_hot = True, returns the onehot of the categorical columns, by means of self.columns_to_onehot
        """
        if len(self.cache) == 0:
            path = 'resources/dataset/preprocessed/{}/features/{}/features.csv.gz'.format(self.mode, self.name)
            if not os.path.exists(path):
                choice = yesno_choice('feature \'{}\' does not exist. want to create?'.format(self.name))
                if choice == 'y':
                    self.save_feature()
                else:
                    return

            df = pd.read_csv(path, index_col=None, engine='c')
            self.cache = df
            print('{} feature read and cached'.format(self.name))
        
        else:
            df = self.cache
            print('{} feature read from cache'.format(self.name))

        # then proceed with one hot
        if one_hot:
            for t in self.columns_to_onehot:
                col = df[t[0]]
                if t[1] == 'single':
                    one_hot_prefix = t[2] if len(t) == 3 else t[0]
                    oh = pd.get_dummies(col, prefix=one_hot_prefix)
                elif t[1] == 'multiple':
                    mid = col.apply(lambda x: x.split('|') if isinstance(x, str) else x)
                    mid.fillna(value='', inplace=True)
                    mlb = MultiLabelBinarizer()
                    oh = mlb.fit_transform(mid)
                    oh = pd.DataFrame(oh, columns=mlb.classes_)
                    oh = oh.astype(np.uint8)
                    oh = oh.add_prefix(one_hot_prefix)

                df = df.drop([t[0]], axis=1)
                df = pd.concat([df, oh], axis=1)
            
            print('{} onehot completed'.format(self.name))

        df = self.post_loading(df)
        return df