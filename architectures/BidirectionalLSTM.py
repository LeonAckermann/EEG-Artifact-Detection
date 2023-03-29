import sys
sys.path.append('./')
from mylibs import *

class BidirectionalLSTM(keras.Model):

    def __init__(self, 
                 num_hidden_units, 
                 num_bidirectional_layers, 
                 num_dense_units=0,
                 num_dense_layers=0,
                 activation_function='tanh',
                 increase=False):
        
        
        super(BidirectionalLSTM, self).__init__()


        self.lstm_layers_list = []
        for i in range(num_bidirectional_layers):
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
        for layer in self.lstm_layers_list:
            x = layer(x)

        if self.num_dense_layers > 0:
            for layer in self.dense_layers_list:
                x = layer(x)

        return self.dense(x)