import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
scaler = MinMaxScaler(feature_range=(0, 1))

class PreProcessing:
    def __init__(self, df, endog_var):
        self.data:pd.DataFrame = df
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.endog_var = endog_var
        self.scaler_y = MinMaxScaler(feature_range=(0, 1))

    def clean (self, dropped_columns):
        self.data = self.data.dropna()
        self.data = self.data.reset_index()
        self.data = self.data.drop(columns=dropped_columns)

    def shift(self, n_in, n_out):
        n_vars = 1 if type(self.data) is list else self.data.shape[1]
        cols, names = list(), list()

        for i in range(n_in, 0, -1):
            cols.append(self.data.shift(i))
            names += [(self.data.columns[j]+'(t-%d)' % (i)) for j in range(n_vars)]

        for i in range(0, n_out):
            cols.append(self.data[self.endog_var].shift(-i))
            if i == 0:
                names += [(self.endog_var+'(t)')]
            else:
                names += [(self.endog_var+'(t+%d)' % (i))]

        agg = pd.concat(cols, axis=1)
        agg.columns = names
        self.data = agg
        self.data.dropna(inplace=True)
        self.data.reset_index(inplace=True)
        self.data.pop('index')

    def get_train_test_normalized(self, train_size, y):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.data.drop(columns=[y]),
                                                                               self.data[y], train_size=train_size)
        self.X_train = scaler.fit_transform(self.X_train)
        self.X_test = scaler.fit_transform(self.X_test)
        self.y_train = scaler.fit_transform(self.y_train.values.reshape(-1,1))
        self.y_test = self.scaler_y.fit_transform(self.y_test.values.reshape(-1,1))

        #return self.X_train, self.X_test, self.y_train, self.y_test

    def reshape_LSTM(self):
        self.X_test = self.X_test.reshape((self.X_test.shape[0], 1, self.X_test.shape[1]))
        self.X_train = self.X_train.reshape((self.X_train.shape[0], 1, self.X_train.shape[1]))

    def inverse_LSTM(self, prediction):
        self.y_test = self.scaler_y.inverse_transform(self.y_test)
        prediction = self.scaler_y.inverse_transform(prediction)
        return self.y_test, prediction