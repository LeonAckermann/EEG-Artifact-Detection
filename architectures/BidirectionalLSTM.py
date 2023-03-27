import sys
sys.path.append('./')
from mylibs import *

class BidirectionalLSTM(keras.Model):

    def __init__(self, hidden_units, number_bidirectional_layers, activation_function='tanh'):
        super(BidirectionalLSTM, self).__init__()

        self.lstm_layers_list = [
            keras.layers.Bidirectional(keras.layers.LSTM(hidden_units, 
                                                         return_sequences=True,
                                                         activation=activation_function))
            for i in range(number_bidirectional_layers)
        ]
      
        self.dense = keras.layers.Dense(2, activation="sigmoid")

    def call(self, inputs, training=False):
        x = inputs
        for layer in self.lstm_layers_list:
            x = layer(x)

        return self.dense(x)