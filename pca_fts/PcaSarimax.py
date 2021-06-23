import pandas as pd
from pca_fts.PcaTransformation import PcaTransformation
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.statespace.sarimax import SARIMAX

class PcaSarimax():
    def __init__(self, n_components, endogen_variable,order,seasonal_order):

        self.n_components = n_components;
        self.endogen_variable = endogen_variable;
        self.model = None
        self.order = order
        self.seasonal_order = seasonal_order

    def run_train_model(self,data):
        pca_reduced = self.create_sarimax(data)
        sarimax = self.fit_sarimax()
        return self.model, sarimax, pca_reduced

    def run_test_model(self,data, sarimax,start,end):
        pca_reduced = self.apply_pca(data)
        exog = data.drop(labels=[self.endogen_variable], axis=1)
        forecast = self.forecast_sarimax(sarimax = sarimax, start = start, end=end, exog = exog)
        return forecast, pca_reduced

    def apply_pca(self, data):
        scaled = MinMaxScaler()

        target = self.endogen_variable
        if target not in data.columns:
            target = None
        cols = data.columns[:-1] if target is None else [col for col in data.columns if
                                                         col != target]

        scaled_data = scaled.fit_transform(data[cols].values)
        scaled_data = pd.DataFrame(columns=data.columns[:-1], data=scaled_data)
        scaled_data[target] = data[target].values

        pca = PcaTransformation()
        pca_reduced = pca.apply(data=scaled_data, endogen_variable=self.endogen_variable)
        return (pca_reduced)

    def create_sarimax(self, data):
        reduced = self.apply_pca(data)
        exog = data.drop(labels=[self.endogen_variable], axis=1)
        self.model = SARIMAX(endog = list(reduced[self.endogen_variable]),
                        exog = exog,
                        order = self.order,
                        seasonal_order = self.seasonal_order,
                        enforce_invertibility=False,
                        enforce_stationarity=False)
        return reduced

    def fit_sarimax(self):
        return self.model.fit()

    def forecast_sarimax(self, sarimax, start, end, exog):
        return sarimax.predict(start=start, end= end, exog=exog)
