import pandas as pd
from AETransformation import AutoencoderTransformation
from sklearn.preprocessing import MinMaxScaler
from pyFTS.models.multivariate.variable import Variable
from pyFTS.partitioners.Grid import GridPartitioner
from pyFTS.models.multivariate.wmvfts import WeightedMVFTS


class AEWeightedMVFTS():
    def __init__(self, n_dimensions,endogen_variable, n_part, names):

        self.n_dimensions = n_dimensions
        self.endogen_variable = endogen_variable
        self.n_part = n_part
        self.model = None
        self.names = names

    def run_train_model(self,data=None, epochs=50, n_layers=2, neuron_per_layer = None, batch = 40,
                opt = 'adam', act = 'tanh'):
        ae_reduced = self.create_wmvfts(data, epochs, n_layers, neuron_per_layer, batch, opt, act)
        self.fit_wmvfts(ae_reduced)
        return self.model, ae_reduced

    def run_test_model(self,model=None,data=None, epochs=50, n_layers=2, neuron_per_layer = None, batch = 40,
                opt = 'adam', act = 'tanh'):
        ae_reduced = self.apply_ae(data, epochs, n_layers, neuron_per_layer, batch, opt, act)
        forecast, forecast_self = self.forecast_wmvfts(model,ae_reduced)
        return forecast, forecast_self, ae_reduced

    def apply_ae(self,data=None, epochs=50, n_layers=2, neuron_per_layer = None, batch = 40,
                opt = 'adam', act = 'tanh'):

        target = self.endogen_variable
        if target not in data.columns:
            target = None
        AE = AutoencoderTransformation(reduced_dimension = self.n_dimensions)
        ae_reduced = AE.apply(data=data, epochs=epochs, n_layers=n_layers, neuron_per_layer=neuron_per_layer,
                              batch=batch, opt=opt, act=act, endogen_variable=self.endogen_variable, names=self.names)
        return ae_reduced
     
    def create_wmvfts(self,data=None, epochs=50, n_layers=2, neuron_per_layer = None, batch = 40,
                opt = 'adam', act = 'tanh'):
        reduced = self.apply_ae(data=data, epochs=epochs, n_layers=n_layers, neuron_per_layer=neuron_per_layer,
                               batch=batch, opt=opt, act=act)

        exog = []
        dims = reduced.columns

        for i in range(0,self.n_dimensions):
            exog.append(Variable("v"+str(i), data_label=dims[i], partitioner=GridPartitioner, npart=self.n_part, data=reduced))

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

