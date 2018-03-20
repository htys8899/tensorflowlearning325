# coding=utf-8
# import time
import tensorflow as tf
import tensorflow.contrib.eager as tfe

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

class Model(object):
    def __init__(self):
        self.layer = tf.layers.Dense(1)

    def predict(self, inputs):
        return self.layer(inputs)

class MNISTModel(object):
    def __init__(self, data_format):
    # 'channels_first' is typically faster on GPUs
    # while 'channels_last' is typically faster on CPUs.
    # See: https://www.tensorflow.org/performance/performance_guide#data_formats
        if data_format == 'channels_first':
            self._input_shape = [-1, 1, 28, 28]
        else:
            self._input_shape = [-1, 28, 28, 1]
        self.conv1 = tf.layers.Conv2D(32, 5,
                                  padding='same',
                                  activation=tf.nn.relu,
                                  data_format=data_format)
        self.max_pool2d = tf.layers.MaxPooling2D(
        (2, 2), (2, 2), padding='same', data_format=data_format)
        self.conv2 = tf.layers.Conv2D(64, 5,
                                  padding='same',
                                  activation=tf.nn.relu,
                                  data_format=data_format)
        self.dense1 = tf.layers.Dense(1024, activation=tf.nn.relu)
        self.dropout = tf.layers.Dropout(0.5)
        self.dense2 = tf.layers.Dense(10)

    def predict(self, inputs):
        x = tf.reshape(inputs, self._input_shape)
        x = self.max_pool2d(self.conv1(x))
        x = self.max_pool2d(self.conv2(x))
        x = tf.layers.flatten(x)
        x = self.dropout(self.dense1(x))
        return self.dense2(x)

def loss(model, inputs, targets):
    return tf.reduce_mean(
      tf.nn.softmax_cross_entropy_with_logits(
          logits=model.predict(inputs), labels=targets))


# Load the training and validation data
from tensorflow.examples.tutorials.mnist import input_data
data = input_data.read_data_sets("./mnist_data", one_hot=True)

# Train
device = "gpu:0" if tfe.num_gpus() else "cpu:0"
model = MNISTModel('channels_first' if tfe.num_gpus() else 'channels_last')
optimizer = tf.train.AdamOptimizer(learning_rate=1e-4)
grad = tfe.implicit_gradients(loss)
for i in range(20001):
    with tf.device(device):
        (inputs, targets) = data.train.next_batch(50)
    optimizer.apply_gradients(grad(model, inputs, targets))
    if i % 100 == 0:
      print("Step %d: Loss on training set : %f" %
            (i, loss(model, inputs, targets).numpy()))
print("Loss on test set: %f" % loss(model, data.test.images, data.test.labels).numpy())