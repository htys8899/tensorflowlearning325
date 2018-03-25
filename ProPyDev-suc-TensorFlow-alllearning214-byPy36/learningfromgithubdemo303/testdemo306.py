# coding=utf-8
#import time
import tensorflow as tf
#import tensorflow.contrib.eager as tfe
#import numpy as np
#import tensorflow as tf
#import numpy
import matplotlib.pyplot as plt

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


import numpy as np

np_x = np.array(2., dtype=np.float32)
x = tf.constant(np_x)

py_y = 3.
y = tf.constant(py_y)

z = x + y + 1

print(z)
print(z.numpy())