# coding=utf-8
import time
import tensorflow as tf
import tensorflow.contrib.eager as tfe

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def measure(x):
    # The very first time a GPU is used by TensorFlow, it is initialized.
    # So exclude the first run from timing.
    tf.matmul(x, x)

    start = time.time()
    for i in range(10):
        tf.matmul(x, x)
    end = time.time()
    return "Took %s seconds to multiply a %s matrix by itself 10 times" % (end - start, x.shape)

# Run on CPU:
with tf.device("/cpu:0"):
    print("CPU: %s" % measure(tf.random_normal([1000, 1000])))

# If a GPU is available, run on GP

if tfe.num_gpus() > 0:
    with tf.device("/gpu:0"):
        print("GPU: %s" % measure(tf.random_normal([1000, 1000])))

x = tf.random_normal([10, 10])

x_gpu0 = x.gpu()
x_cpu = x.cpu()

_ = tf.matmul(x_cpu, x_cpu)  # Runs on CPU
_ = tf.matmul(x_gpu0, x_gpu0)  # Runs on GPU:0

if tfe.num_gpus() > 1:
    x_gpu1 = x.gpu(1)
    _ = tf.matmul(x_gpu1, x_gpu1)  # Runs on GPU:1

# ���裬�����tfe��һ��num_gpus�����أ� ��ΰ�����



