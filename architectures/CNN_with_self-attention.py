import sys
sys.path.append('./')
from mylibs import *

# CCNAttentionNetwork is a custom Keras model that combines a 1D convolutional neural network (CNN)
# with multi-head attention mechanism and fully connected (dense) layers.

class CCNAttentionNetwork(tf.keras.Model):

      """
      The constructor takes the following arguments:
      num_units: Integer, the number of units in each dense layer.
      num_layers: Integer, the number of dense layers in the model.
      num_heads: Integer, the number of attention heads in the MultiHeadAttention layer.
      num_conv_layers: Integer, the number of convolutional and max-pooling layer pairs in the conv_block.
      attention: Boolean, this parameter controls whether multihead-attention is used.
      """
      
    def __init__(self, num_units, num_layers, num_heads, num_conv_layers, attention):

        # Call the constructor of the parent class (tf.keras.Model)
        super(CCNAttentionNetwork, self).__init__()

        # Initialize a Keras Sequential model for the convolutional block
        self.conv_block = tf.keras.Sequential()

        # Add the specified number of Conv1D and MaxPooling1D layers to the conv_block
        for i in range(num_conv_layers):
            self.conv_block.add(tf.keras.layers.Conv1D(8 * i + 8, 3, activation='relu'))
            self.conv_block.add(tf.keras.layers.MaxPooling1D(2))

        # Initialize a Keras Sequential model for the dense layers
        self.dense_layers = tf.keras.Sequential()

        # Add the specified number of dense layers with the given number of units and ReLU activation
        for i in range(num_layers):
            self.dense_layers.add(tf.keras.layers.Dense(num_units, activation='relu'))

        # Initialize the attention attribute (not used in this implementation)
        self.attention = None

        # Initialize a MultiHeadAttention layer with the specified number of heads and key/query/value dimensions
        self.mha = tf.keras.layers.MultiHeadAttention(num_heads, 640)

        # Initialize a Dropout layer with a dropout rate of 0.3
        self.dropout = tf.keras.layers.Dropout(0.3)

        # Initialize a Dense layer with 640 units and a sigmoid activation function
        self.dense = tf.keras.layers.Dense(640, activation="sigmoid")
   

  def call(self, x):
    
    
    x = self.conv_block(x)
                
    if self.attention == True:
        
        x = self.mha(x, x)
        x = self.dropout(x)
    
    
    x = self.dense_layers(x)
        
   
    output_muscle = self.dense(x)

    
    
    
    output_muscle = tf.math.reduce_mean(output_muscle, axis=1)

    return output_muscle


