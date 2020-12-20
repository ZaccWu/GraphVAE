import numpy as np
import tensorflow as tf

"""
Disclaimer: the weight_variable_glorot function from this file comes from 
tkipf/gae original repository on Graph Autoencoders
"""

def weightVariableGlorot(inputDim, outputDim, name =""):
    """
    Create a weight variable with Glorot&Bengio (AISTATS 2010) initialization
    """
    initRange = np.sqrt(6.0 / (inputDim + outputDim))
    initial = tf.random_uniform([inputDim, outputDim], minval = -initRange,
                                maxval = initRange, dtype = tf.float32)
    return tf.Variable(initial, name = name)