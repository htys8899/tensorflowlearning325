# coding=utf-8
import time
import tensorflow as tf
import tensorflow.contrib.eager as tfe

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

class TwoLayerNet(tfe.Network):
    def __init__(self):
        super(TwoLayerNet, self).__init__()
        self.layer1 = self.track_layer(tf.layers.Dense(2, activation=tf.nn.relu, use_bias=False))
        self.layer2 = self.track_layer(tf.layers.Dense(3, use_bias=False))

    def call(self, x):
        return self.layer2(self.layer1(x))

net = TwoLayerNet()

# No variables created yet
assert 0 == len(net.variables)

# They are created on first input:
inp = tf.constant([[1.]])

# Since input is a 1x1 matrix, net.l1 has 2 units and net.l2 has 3 units,
# the output is the product of a 1x1 matrix with a 1x2 matrix with a 2x3
# matrix.
assert [1, 3] == net(inp).shape.as_list()  # Invoke net; get output shape.
assert 1 == len(net.layer1.variables)
assert 1 == len(net.layer2.variables)
assert 2 == len(net.variables)  # weights for each layer.
assert [1, 2] == net.variables[0].shape.as_list()  # weights of layer1.
assert [2, 3] == net.variables[1].shape.as_list()  # weights of layer2.
print("20180303--88")

class ThreeLayerNet(tfe.Network):
    def __init__(self):
        super(ThreeLayerNet, self).__init__()
        self.a = self.track_layer(TwoLayerNet())
        self.b = self.track_layer(tf.layers.Dense(4, use_bias=False))

    def call(self, x):
        return self.b(self.a(x))

net = ThreeLayerNet()

assert [1, 4] == net(inp).shape.as_list()
assert 3 == len(net.variables)
assert [1, 2] == net.variables[0].shape.as_list()
assert [2, 3] == net.variables[1].shape.as_list()
assert [3, 4] == net.variables[2].shape.as_list()

print ("20180304-ok")
