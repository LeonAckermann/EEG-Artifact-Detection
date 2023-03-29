import sys
sys.path.append('./')
from mylibs import *

class MultiLayerLSTM(keras.Model):

    def __init__(self, 
                 num_hidden_units, 
                 num_lstm_layers,
                 num_dense_units=0,
                 num_dense_layers=0,
                 num_conv_layers=0,
                 activation_function='tanh',
                 increase=0):
        
        super(MultiLayerLSTM, self).__init__()

        self.conv_layers_list = []
        self.num_conv_layers = num_conv_layers
        if num_conv_layers > 0:
            for i in range(num_conv_layers):
                self.conv_layers_list.append(layers.TimeDistributed(layers.Conv1D(filters=8*i+8, kernel_size=3, activation='relu')))
            self.conv_layers_list.append(layers.TimeDistributed(layers.MaxPool1D()))
            self.conv_layers_list.append(layers.TimeDistributed(layers.Flatten()))

        self.lstm_layers_list = []
        for i in range(num_lstm_layers):
            self.lstm_layers_list.append(keras.layers.LSTM(num_hidden_units, 
                                                           return_sequences=True, 
                                                           activation=activation_function))
      
        self.dense_layers_list = []
        self.num_dense_layers = num_dense_layers
        if num_dense_layers > 0:
            for i in range(num_dense_layers):
                self.dense_layers_list.append(keras.layers.Dense(num_dense_units, activation='relu'))

        self.dense = keras.layers.Dense(2, activation="sigmoid")

    def call(self, inputs, training=False):
        x = inputs

        if self.num_conv_layers > 0:
            for layer in self.conv_layers_list:
                x = layer(x)

        for layer in self.lstm_layers_list:
            x = layer(x)
        
        if self.num_dense_layers > 0:
            for layer in self.num_dense_layers:
                x = layer(x)
        
        return self.dense(x)
      