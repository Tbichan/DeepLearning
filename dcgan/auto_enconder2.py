# -*- coding: utf-8 -*-

# モジュールのインポート
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import cv2
import matplotlib.pyplot as plt

H = 50
BATCH_SIZE = 100
EPOCH = 1000

#MNISTデータの読み込み
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

# エンコーダー
class Encoder(object):
    def __init__(self, input, n_in, n_out, vs_enc='encoder'):
        self.input = input
        with tf.variable_scope(vs_enc):
            weight_init= tf.truncated_normal_initializer(mean=0.0, stddev=0.05)
            W = tf.get_variable('W', [n_in, n_out], initializer=weight_init)
            bias_init = tf.constant_initializer(value=0.0)
            b = tf.get_variable('b', [n_out], initializer=weight_init)
        self.w = W
        self.b = b

    def output(self):
        linarg = tf.matmul(self.input, self.w) + self.b
        self.output = tf.nn.relu(linarg)
        print(self.output)
        return self.output

# デコーダー
class Decoder(object):
    def __init__(self, input, n_in, n_out, vs_dec='decoder'):
        self.input = input

        if vs_dec == 'decoder':
            with tf.variable_scope(vs_dec):
                weight_init= tf.truncated_normal_initializer(mean=0.0, stddev=0.05)
                W = tf.get_variable('W', [n_in, n_out], initializer=weight_init)
            
        else:   # 重み共有
            with tf.variable_scope(vs_dec, reuse=True):
                W = tf.get_variable('W', [n_out, n_in])
                W = tf.transpose(W)

        with tf.variable_scope('decoder'):
            bias_init = tf.constant_initializer(value=0.0)
            b = tf.get_variable('b', [n_out], initializer=bias_init)
        
        self.w = W
        self.b = b

    def output(self):
        linarg = tf.matmul(self.input, self.w) + self.b
        self.output = tf.nn.relu(linarg)

        return self.output

    
x = tf.placeholder("float", [None, 784])
#y_  = tf.placeholder("float", [None, 784])

def make_model(x):
    enc_layer = Encoder(x, 784, 625, vs_enc='encoder')
    enc_out = enc_layer.output()
    # 重み共有
    dec_layer = Decoder(enc_out, 625, 784, vs_dec='encoder')
    dec_out = dec_layer.output()
    return enc_out, dec_out

encoded, decoded = make_model(x)
loss = tf.reduce_mean((x-decoded)*(x-decoded))
optimizer = tf.train.AdamOptimizer(0.01).minimize(loss)

# Train
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    for i in range(10001):
        batch_xs, batch_ys = mnist.train.next_batch(128)
        optimizer.run({x: batch_xs})

        if i % 1000 == 0:
            train_loss = loss.eval({x: batch_xs})
            print('  step, loss = %6d: %6.3f' % (i, train_loss))
        
    # generate decoded image with test data
    test_fd = {x: mnist.test.images}
    decoded_imgs = decoded.eval(test_fd)
    print('loss (test) = ', loss.eval(test_fd))

x_test = mnist.test.images

n = 10  # how many digits we will display
plt.figure(figsize=(20, 4))
for i in range(n):
    # display original
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(x_test[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # display reconstruction
    ax = plt.subplot(2, n, i + 1 + n)
    plt.imshow(decoded_imgs[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

plt.savefig('mnist_ae1.png')
