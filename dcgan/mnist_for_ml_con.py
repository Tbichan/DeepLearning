# -*- coding: utf-8 -*-

# モジュールのインポート
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
 

#MNISTデータの読み込み
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

 

#セッションの作成
sess = tf.InteractiveSession()
 
# モデルの作成
def weight_variable(shape):
    initial = tf.random_normal(shape, mean=0.0, stddev=0.05)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.zeros(shape=shape)
    return tf.Variable(initial)

# 畳み込み層
def conv2d(x, W):
    print(x)
    return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME')

# プーリング層
def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')

"""padding='SAME' は(フィルタサイズ-1)分0パディング"""
    
x = tf.placeholder(tf.float32, [None, 784])

""" 第1レイヤー
5,5はフィルタサイズ、1は入力チャンネル数、30はフィルタ数
"""
W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])
x_image = tf.reshape(x, [-1, 28, 28, 1])
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

# 第2レイヤー
W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

# 全結合層
W_fc1 = weight_variable([7 * 7 * 64, 100])
b_fc1 = bias_variable([100])
# 整形
h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

# Dropout
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# 出力層
W_fc2 = weight_variable([100, 10])
b_fc2 = bias_variable([10])
y = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

# 学習モデルを作成(損失とオプティマイザーを定義)
y_ = tf.placeholder(tf.float32, [None, 10])
cross_entropy = -tf.reduce_sum(y_ * tf.log(y))

# L2正規化
L2_sqr = tf.nn.l2_loss(W_conv1) + tf.nn.l2_loss(W_conv2)\
+ tf.nn.l2_loss(W_fc1) + tf.nn.l2_loss(W_fc2)
lambda_2 = 0.01

loss = cross_entropy + 0.5 * lambda_2 * L2_sqr
train_step = tf.train.AdamOptimizer(0.0001).minimize(loss)

 

# イニシャライズ
tf.global_variables_initializer().run()

 
# 学習
for i in range(10000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    #print(batch_ys)
    train_step.run({x: batch_xs, y_: batch_ys, keep_prob: 0.5})
    if i % 100 == 0:
        loss_val=sess.run(loss,{x: batch_xs, y_: batch_ys, keep_prob: 1.0})
        print("Step:%d, Loss:%f"%(i,loss_val))


# 学習済みモデルにテストデータを入れてテスト
for i in range(1):
    print(sess.run(tf.argmax(y, 1), feed_dict={x: [mnist.test.images[i]], keep_prob: 1.0}))
    print(mnist.test.labels[i])

# 重み出力
"""
w_wal=sess.run(W1)
b_val=sess.run(b1)
print(w_wal)
print(b_val)
"""

#学習済みモデルの正解率表示
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print(accuracy.eval({x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))
