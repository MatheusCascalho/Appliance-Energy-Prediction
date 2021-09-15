import pandas as pd
import numpy as np

class PreProcessing:
    def __init__(self, df):
        self.data:pd.DataFrame = df

    def clean (self, dropped_columns):
        self.data = self.data.dropna()
        self.data = self.data.reset_index()
        self.data = self.data.drop(columns=dropped_columns)
        return self.data

    # convert series to supervised learning
    def shift(self, n_in, n_out, endog_var):
        n_vars = 1 if type(self.data) is list else self.data.shape[1]
        cols, names = list(), list()

        for i in range(n_in, 0, -1):
            cols.append(self.data.shift(i))
            names += [(self.data.columns[j]+'(t-%d)' % (i)) for j in range(n_vars)]

        for i in range(0, n_out):
            cols.append(self.data[endog_var].shift(-i))
            if i == 0:
                names += [(endog_var+'(t)')]
            else:
                names += [(endog_var+'(t+%d)' % (i))]

        agg = pd.concat(cols, axis=1)
        agg.columns = names
        self.data = agg
        return self.data
