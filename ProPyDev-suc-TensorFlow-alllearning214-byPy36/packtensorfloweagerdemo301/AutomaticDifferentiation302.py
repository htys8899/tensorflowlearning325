# coding=utf-8
import time
import tensorflow as tf
import tensorflow.contrib.eager as tfe

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def f(x):
    return tf.multiply(x, x)  # Or x * x
assert 9 == f(3.).numpy()

df = tfe.gradients_function(f)
assert 6 == df(3.)[0].numpy()

# Second order deriviative.
d2f = tfe.gradients_function(lambda x: df(x)[0])
assert 2 == d2f(3.)[0].numpy()

# Third order derivative.
d3f = tfe.gradients_function(lambda x : d2f(x)[0])
assert 0 == d3f(3.)[0].numpy()


def prediction(input, weight, bias):
    return input * weight + bias

# A toy dataset of points around 3 * x + 2
NUM_EXAMPLES = 1000
training_inputs = tf.random_normal([NUM_EXAMPLES])
noise = tf.random_normal([NUM_EXAMPLES])
training_outputs = training_inputs * 3 + 2 + noise

# A loss function: Mean-squared error
def loss(weight, bias):
    error = prediction(training_inputs, weight, bias) - training_outputs
    return tf.reduce_mean(tf.square(error))

# Function that returns the derivative of loss with respect to
# weight and bias
grad = tfe.gradients_function(loss)

# Train for 200 steps (starting from some random choice for W and B, on the same
# batch of data).
W = 5.
B = 10.
learning_rate = 0.01
print("Initial loss: %f" % loss(W, B).numpy())
for i in range(200):
    (dW, dB) = grad(W, B)
    W -= dW * learning_rate
    B -= dB * learning_rate
    if i % 20 == 0:
        print("Loss at step %d: %f" % (i, loss(W, B).numpy()))
print("Final loss: %f" % loss(W, B).numpy())
print("W, B = %f, %f" % (W.numpy(), B.numpy()))

