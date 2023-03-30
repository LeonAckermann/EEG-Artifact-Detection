import sys
sys.path.append('./')
from mylibs import *

class CCNAttentionNetwork(tf.keras.Model):

  def __init__(self, num_units, num_layers, num_heads, num_conv_layers, attention):

    super(CCNAttentionNetwork, self).__init__()
    self.conv_block = tf.keras.Sequential()
    
    for i in range(num_conv_layers):
        
        self.conv_block.add(tf.keras.layers.Conv1D(8*i + 8, 3, activation='relu'))
        self.conv_block.add(tf.keras.layers.MaxPooling1D(2))
        
        
    self.dense_layers = tf.keras.Sequential()
    
    for i in range(num_layers):
        self.dense_layers.add(tf.keras.layers.Dense(num_units, activation='relu'))
    
    
    self.attention = None
    self.mha = tf.keras.layers.MultiHeadAttention(num_heads, 640)
    self.dropout = tf.keras.layers.Dropout(0.3)
    self.dense = tf.keras.layers.Dense(640, activation = "sigmoid")
    self.dense2 = tf.keras.layers.Dense(640, activation = "sigmoid")
    

  def call(self, x):
    
    
    x = self.conv_block(x)
                
    if self.attention == True:
        
        x = self.mha(x, x)
        x = self.dropout(x)
    
    
    x = self.dense_layers(x)
        
    output_eye = self.dense(x)
    output_muscle = self.dense2(x)

    
    
    output_eye = tf.math.reduce_mean(output_eye, axis=1)
    output_muscle = tf.math.reduce_mean(output_muscle, axis=1)

    return output_eye, output_muscle


