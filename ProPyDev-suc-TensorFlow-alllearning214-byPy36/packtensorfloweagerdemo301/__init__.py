import tensorflow as tf

import tensorflow.contrib.eager as tfe
# tfe.enable_eager_execution()
x = tf.matmul([[1, 2],
               [3, 4]],
              [[4, 5],
               [6, 7]])
# Add one to each element
# (tf.add supports broadcasting)
y = tf.add(x, 1)
# Create a random random 5x3 matrix
z = tf.random_uniform([5, 3])
print(x)
print(y)
print(z)

#For convenience, these operations can also be triggered via operator overloading of the Tensor object.
# For example, the +operator is equivalent to tf.add, - to tf.subtract, * to tf.multiply, etc.:
x = (tf.ones([1], dtype=tf.float32) + 1) * 2 - 1
print(x)

import numpy as np
x = tf.add(1, 1)                     # tf.Tensor with a value of 2
y = tf.add(np.array(1), np.array(1)) # tf.Tensor with a value of 2
z = np.multiply(x, y)                # numpy.int64 with a value of 4
print (z)
