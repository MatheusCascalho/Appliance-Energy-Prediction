import pandas as pd
from pca_fts.PcaTransformation import PcaTransformation
from sklearn.preprocessing import MinMaxScaler
from pyFTS.models.multivariate.variable import Variable
from pyFTS.partitioners.Grid import GridPartitioner
from pyFTS.models.multivariate.wmvfts import WeightedMVFTS


class PcaWeightedMVFTS():
    def __init__(self, n_components,endogen_variable, n_part):

        self.n_components = n_components;
        self.endogen_variable = endogen_variable;
        self.n_part = n_part
        self.model = None

    def run_train_model(self,data):
        pca_reduced = self.create_wmvfts(data)
        self.fit_wmvfts(pca_reduced)
        return self.model, pca_reduced

    def run_test_model(self,model,data):
        pca_reduced = self.apply_pca(data)
        forecast, forecast_self = self.forecast_wmvfts(model,pca_reduced)
        return forecast, forecast_self, pca_reduced

    def apply_pca(self,data):
        scaled = MinMaxScaler()

        target = self.endogen_variable
        if target not in data.columns:
            target = None
        cols = data.columns[:-1] if target is None else [col for col in data.columns if
                                                                   col != target]

        scaled_data = scaled.fit_transform(data[cols].values)
        scaled_data = pd.DataFrame(columns=data.columns[:-1], data=scaled_data)
        scaled_data[target] = data[target].values

        pca = PcaTransformation(n_components = self.n_components)
        pca_reduced = pca.apply(data=scaled_data, endogen_variable=self.endogen_variable)
        return pca_reduced

    def create_wmvfts(self,data):
        reduced = self.apply_pca(data)

        exog = []
        components = reduced.columns

        for i in range(0,self.n_components):
            exog.append(Variable("v"+str(i), data_label=components[i], partitioner=GridPartitioner, npart=self.n_part, data=reduced))

        endog = Variable(
            name=self.endogen_variable,
            data_label=self.endogen_variable,
            partitioner=GridPartitioner,
            npart=self.n_part,
            data=reduced
        )

        exog.append(endog)

        self.model = WeightedMVFTS(
            explanatory_variables = exog,
            target_variable = endog
        )
        return reduced

    def fit_wmvfts(self, data):
        self.model.fit(data)

    def forecast_wmvfts(self,model,data):
        result = model.predict(data,steps_ahead=1)
        result_self = self.model.predict(data,steps_ahead=1)
        return result, result_self




