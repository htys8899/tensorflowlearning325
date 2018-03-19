# coding=utf-8

import tensorflow as tf  
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
sess = tf.Session()  
a = tf.constant(103)  
b = tf.constant(114)  
print(sess.run(a + b))