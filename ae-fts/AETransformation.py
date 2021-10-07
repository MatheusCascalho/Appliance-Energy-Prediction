"""
Autoencoders for Fuzzy Time Series
"""

import pandas as pd
import numpy as np
from keras.models import Model
from keras.layers import Dense, Input
from keras import regularizers
from sklearn.preprocessing import MinMaxScaler
from pyFTS.common.transformations.transformation import Transformation


class AutoencoderTransformation(Transformation):
    def __init__(self,
                 reduced_dimension:int = 2,
                 **kwargs):



        # Autoencoder attributes
        self.load_file = kwargs.get('loadFile')
        self.data: pd.DataFrame = None

        self.is_multivariate = True
        self.reduced_dimension = reduced_dimension
        self.encoder_layers = []
        self.decoder_layers = []
        self.input = None
        self.scaler = MinMaxScaler()
        self.names = None
        self.encoder=None


        # debug attributes
        self.name = 'Autoencoders FTS'
        self.shortname = 'AE-FTS'



    def apply_trained_ae(self,data, endogen):

        scaler = MinMaxScaler()
        cols = data.columns[:-1] if endogen is None else [col for col in data.columns if
                                                                   col != endogen]
        data_scaled = scaler.fit_transform(data[cols])
        new_data = pd.DataFrame(self.encoder.predict(data_scaled), columns = self.names)

        return new_data.join(data[endogen].reset_index())







    def train(self,
                  data: pd.DataFrame,
                  percentage_train: float = .7,   
                  epochs: int = 100,
                  n_layers: int = 2,
                  neuron_per_layer: list = [],
                  batch: int=30,
                  opt: str="adam",
                  act: str="tanh"):
        
        self.encoder_layers.clear()
        self.decoder_layers.clear()
        self.data = data
        limit = round(len(self.data) * percentage_train)
        train = self.data[:limit]
        counter = 0
        if (n_layers==1):
            multi_layer = False
        else:
            multi_layer = True
        
        data_scaled = self.scaler.fit_transform(data)
        if (neuron_per_layer == []):
            n = data_scaled.shape[1] - self.reduced_dimension
            aux = (n/n_layers)
            for i in range (1, n_layers):
                neuron_per_layer.append(data_scaled.shape[1] - round(aux*i))

        self.input = Input(shape=(data_scaled.shape[1], ))     
        if (multi_layer):
            self.encoder_layers.append(Dense(neuron_per_layer[0], activation=act, activity_regularizer=regularizers.l1(10e-5))(self.input))
            for i in range (1, n_layers-1):
                self.encoder_layers.append(Dense(neuron_per_layer[i], activation=act)(self.encoder_layers[i-1]))
            self.encoder_layers.append(Dense(self.reduced_dimension, activation=act)(self.encoder_layers[-1]))
            self.decoder_layers.append(Dense(neuron_per_layer[-1], activation=act, activity_regularizer=regularizers.l1(10e-5))(self.encoder_layers[-1]))
            for i in range (n_layers-3, -1, -1):    
                self.decoder_layers.append(Dense(neuron_per_layer[i], activation=act)(self.decoder_layers[counter]))
                counter+=1
            self.decoder_layers.append(Dense(data_scaled.shape[1], activation=act)(self.decoder_layers[counter]))
        else:
            self.encoder_layers.append(Dense(self.reduced_dimension, activation=act, activity_regularizer=regularizers.l1(10e-5))(self.input))
            self.decoder_layers.append(Dense(data_scaled.shape[1], activation=act, activity_regularizer=regularizers.l1(10e-5))(self.encoder_layers[0]))
        autoencoder = Model(self.input, self.decoder_layers[-1])
        autoencoder.compile(optimizer = opt, loss='mse')
        X_train = data_scaled
        autoencoder.fit(x=X_train, y=X_train, epochs=epochs, batch_size=batch)
        
        
    def apply(self,
              data: pd.DataFrame,
              param=None,
              epochs: int = 100,
              n_layers: int = 2,
              neuron_per_layer: list = [],
              batch: int=30,
              opt: str="adam",
              act: str="tanh",
              **kwargs):
        """
        Transform a N-dimensional dataset into a n-dimensional dataset, where one dimension is the endogen variable
        If endogen_variable = None, the last column will be the endogen_variable.
        Args:
            data (pd.DataFrame): N-Dimensional dataset
            endogen_variable (str):  column of dataset
            names (Tuple): names for new columns created by the AutoEncoders Transformation.
            param:
            **kwargs: params of AE's train process
                percentage_train (float). Percentage of dataset that will be used for train SOM network. default: 0.7
                epochs: epochs of SOM network. default: 10000
        """

        endogen_variable = kwargs.get('endogen_variable', None)
        self.names = kwargs.get('names', ('factor_1', 'factor_2'))

        if endogen_variable not in data.columns:
            endogen_variable = None
            

        cols = data.columns[:-1] if endogen_variable is None else [col for col in data.columns if
                                                                   col != endogen_variable]

        data_scaled = self.scaler.fit_transform(data[cols])
                                               
        self.train(data_scaled, 1, epochs, n_layers, neuron_per_layer)
        
        self.encoder = Model(inputs=self.input, outputs=self.encoder_layers[-1])


        new_data = pd.DataFrame(self.encoder.predict(data_scaled), columns = self.names)

        endogen = endogen_variable if endogen_variable is not None else data.columns[-1]
        new_data[endogen] = data[endogen].values
        return new_data