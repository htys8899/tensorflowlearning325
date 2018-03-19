# coding=utf-8

import tensorflow as tf  

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

img = tf.placeholder(name="img", dtype=tf.float32, shape=(1, 64, 64, 3))
val = img + tf.constant([1., 2., 3.]) + tf.constant([1., 4., 4.])
out = tf.identity(val, name="out")
with tf.Session() as sess:
    tflite_model = tf.contrib.lite.toco_convert(sess.graph_def, [img], [out])
    open("converteds_model.tflite", "wb").write(tflite_model)