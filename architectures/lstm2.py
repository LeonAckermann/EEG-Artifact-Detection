import sys
sys.path.append('./')
from mylibs import *

class LSTM2(keras.Model):

    def __init__(self, hidden_units, activation_function: str, dropout: float):
        super(LSTM2, self).__init__()

        self.layers_list = [
            layers.LSTM(hidden_units, return_sequences=True, name="lstm1", activation=activation_function),
            layers.Dropout(rate=dropout, name="dropout1"),
            layers.LSTM(hidden_units, return_sequences=True, name="lstm2", activation=activation_function),
            layers.Dense(2, name="dense1")
        ]
        
    def call(self, inputs, training=False):
        x = inputs
        for layer in self.layers_list:
            x = layer(x)
        return x