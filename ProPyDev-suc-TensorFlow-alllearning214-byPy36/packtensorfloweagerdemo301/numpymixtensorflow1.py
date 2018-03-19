import numpy as np
import tensorflow as tf


np_x = np.array(2., dtype=np.float32)
x = tf.constant(np_x)

py_y = 3.
y = tf.constant(py_y)

z = x + y + 1

print(z)
print(z.numpy())
