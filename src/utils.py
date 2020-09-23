from sklearn.model_selection import train_test_split
from functools import reduce
import pandas as pd
import numpy as np

class MovieLensKFold():
    def __init__(self, n_splits, col_split='userID'):
        self.n_splits = n_splits
        self.col_split = col_split

    def split(self, X, y=None, groups=None):
        for i in range(self.n_splits):
            yield MovieLensKFold.__train_test_split_per_user(X, self.col_split)

    def get_n_splits(self, X, y=None, groups=None):
        return self.n_splits

    @staticmethod
    def __train_test_split_group(X):
        X_train, X_test = train_test_split(X)
        return pd.Series([X_train, X_test], index=['X_train', 'X_test'])

    @staticmethod
    def __train_test_split_per_user(df, col):
        final = df.groupby(col).apply(lambda grp: MovieLensKFold.__train_test_split_group(grp.index.values))
        X_train, X_test = reduce(lambda acc, val: (np.concatenate((acc[0],val[0])),np.concatenate((acc[1],val[1]))), final.values, ([],[]))
        return X_train, X_test
