# coding=utf-8
import time
import tensorflow as tf
import tensorflow.contrib.eager as tfe

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def log1pexp(x):
    return tf.log(1 + tf.exp(x))

grad_log1pexp = tfe.gradients_function(log1pexp)

# Works fine at x = 0.
assert 0.5 == float(grad_log1pexp(0.)[0])

# Returns a `nan` at x = 100 due to numerical instability.
import math
assert math.isnan(float(grad_log1pexp(100.)[0]))

@tfe.custom_gradient
def log1pexp(x):
    grad_log1pexp = tfe.gradients_function(log1pexp)

# Works as before at x = 0.
assert 0.5 == float(grad_log1pexp(0.)[0])

# But now works at x = 100 as well.
assert 1.0 == float(grad_log1pexp(100.)[0])
