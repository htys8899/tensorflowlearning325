# coding=utf-8

import tensorflow as tf  

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

x = tf.placeholder(tf.float32, shape=[1, 1])
m = tf.matmul(x, x)
print(m) 

with tf.Session() as sess:
    print(sess.run(m, feed_dict={x: [[2.]]}))
# Will print [[4.]]

#20180228
#Eager execution makes this much simpler:

x = [[2.]]
m = tf.matmul(x, x)
print(m) 

a = tf.constant([1, 2, 3, 4, 5, 6], shape=[2, 3])
b = tf.constant([7, 8, 9, 10, 11, 12], shape=[3, 2])
c = tf.matmul(a, b)
print(c)

# With TensorFlow installed, eager execution is enabled via a single call:

# import tensorflow as tf

import tensorflow.contrib.eager as tfe

#tfe.enable_eager_execution()
