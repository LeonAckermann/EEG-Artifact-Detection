import sys
sys.path.append('./')
from mylibs import *

class LSTM1(keras.Model):

    def __init__(self, hidden_units, activation_function, output_bias=None):
        super(LSTM1, self).__init__()

        self.layers_list = [
            keras.layers.LSTM(hidden_units, return_sequences=True, activation=activation_function, name="lstm1"),
            keras.layers.Dense(2, activation='sigmoid', bias_initializer=output_bias, name="lstm2") 
        ]

    def call(self, inputs, training=False):
        x = inputs
        for layer in self.layers_list:
            x = layer(x)

        return x