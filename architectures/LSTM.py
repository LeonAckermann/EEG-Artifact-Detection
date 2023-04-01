import sys
sys.path.append('./')
from mylibs import *

class LSTM(keras.Model):

    """
      The constructor takes the following arguments:
      num_hidden_units: Integer, the number of hidden units in lstm layer
      num_lstm_layers: Integer, the number of LSTM layers
      num_dense_units: Integer, the number of units in each dense layer.
      num_dense_layers: Integer, the number of dense layers in the model.
      num_conv_layers: Integer, the number of convolutional layers in the conv_block.
      increase: bool, whether hidden_units increase with in higher lstm layers
      bidirectiona: Boolean, whether bidirectional or multilayer lstm used
      """

    def __init__(self, 
                 num_hidden_units, 
                 num_lstm_layers, 
                 num_dense_units=0,
                 num_dense_layers=0,
                 num_conv_layers=0,
                 increase=0,
                 bidirectional=False):
        
        
        super(LSTM, self).__init__()

        self.conv_layers_list = []
        self.num_conv_layers = num_conv_layers
        if num_conv_layers > 0:
            for i in range(num_conv_layers):
                self.conv_layers_list.append(layers.TimeDistributed(layers.Conv1D(filters=8*i+8, kernel_size=3, activation='relu')))
            self.conv_layers_list.append(layers.TimeDistributed(layers.MaxPool1D()))
            self.conv_layers_list.append(layers.TimeDistributed(layers.Flatten()))

        
        self.lstm_layers_list = []
        if bidirectional:
            for i in range(num_lstm_layers):
                self.lstm_layers_list.append(keras.layers.Bidirectional(keras.layers.LSTM(num_hidden_units+(num_hidden_units*i*increase), 
                                                                                      return_sequences=True,
                                                                                      activation='tanh')))
        else:
            for i in range(num_lstm_layers):
                self.lstm_layers_list.append(keras.layers.LSTM(num_hidden_units+(num_hidden_units*i*increase), 
                                                               return_sequences=True, 
                                                               activation='tanh'))
        self.dropout = keras.layers.Dropout(0.5)
        
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
                print(x)
                x = layer(x)
    
        for layer in self.lstm_layers_list:
            x = layer(x)
        
        x = self.dropout(x)

        if self.num_dense_layers > 0:
            for layer in self.dense_layers_list:
                x = layer(x)

        return self.dense(x)