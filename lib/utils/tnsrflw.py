import numpy as np
import tensorflow as tf

@tf.function(jit_compile=True)
def logit(val:tf.float32):
    return tf.math.log(val) - tf.math.log(1.0 - val)

@tf.function(jit_compile=True)
def sigmoid_transform(val:tf.float32, lb:tf.float32, ub:tf.float32):
    return lb + (ub - lb) * tf.math.sigmoid(val)

@tf.function(jit_compile=True)
def inv_sigmoid_transform(val:tf.float32, lb:tf.float32, ub:tf.float32):
    return logit((val - lb)/(ub - lb))

