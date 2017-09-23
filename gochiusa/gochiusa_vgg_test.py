#coding:utf-8

import cv2
import numpy as np
import tensorflow as tf
import os
import math

INPUT_W = 28
INPUT_H = 28
INPUT_C = 3
INPUT_NUM = INPUT_W*INPUT_H*INPUT_C
TRAIN_NUM = 75
CATEGORY_NUM = 5
TRAIN_ONE_NUM = int(TRAIN_NUM / CATEGORY_NUM)

BATCH_SIZE = 75

HIDE = 4096

# seed値固定
tf.set_random_seed(0)

#セッションの作成
sess = tf.InteractiveSession()
 
# モデルの作成
def weight_variable(shape, stddev=0.05):

    # Heの初期値
    n=shape[0]
    stddev = tf.sqrt(2.0/n)
    initial = tf.random_normal(shape, stddev=stddev)
    return tf.Variable(initial)

def weight_variable_conv(shape):
    kW = shape[0]
    kH = shape[1]
    outPlane = shape[3]
    n=kW*kH*outPlane
    stddev = tf.sqrt(2.0/n)
    initial = tf.random_normal(shape, stddev=stddev)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.zeros(shape=shape)
    return tf.Variable(initial)

# 畳み込み層
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME')

# Batch Normalization
def batch_normalization(shape, input):
    gamma = weight_variable([shape])
    beta = weight_variable([shape])
    mean, variance = tf.nn.moments(input, [0])
    return gamma * (input - mean) / tf.sqrt(variance + 1e-5) + beta

# プーリング層
def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')

"""padding='SAME' は(フィルタサイズ-1)分0パディング"""
    
x = tf.placeholder(tf.float32, [None, INPUT_NUM])

""" 第1レイヤー
5,5はフィルタサイズ、3は入力チャンネル数、64はフィルタ数
"""
W_conv1 = weight_variable_conv([5, 5, INPUT_C, 64])
b_conv1 = bias_variable([64])
x_image = tf.reshape(x, [-1, INPUT_W, INPUT_H, INPUT_C])
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)

W_conv2 = weight_variable_conv([5, 5, 64, 64])
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_conv1, W_conv2) + b_conv2)
#bn2 = batch_normalization(64, h_conv2)
#h_pool2 = max_pool_2x2(bn2)
h_pool2 = max_pool_2x2(h_conv2)


W_conv3 = weight_variable_conv([3, 3, 64, 128])
b_conv3 = bias_variable([128])
h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3) + b_conv3)

W_conv4 = weight_variable_conv([3, 3, 128, 128])
b_conv4 = bias_variable([128])
h_conv4 = tf.nn.relu(conv2d(h_conv3, W_conv4) + b_conv4)
#bn4 = batch_normalization(128, h_conv4)
#h_pool4 = max_pool_2x2(bn4)
h_pool4 = max_pool_2x2(h_conv4)


W_conv5 = weight_variable_conv([3, 3, 128, 256])
b_conv5 = bias_variable([256])
h_conv5 = tf.nn.relu(conv2d(h_pool4, W_conv5) + b_conv5)

W_conv6 = weight_variable_conv([3, 3, 256, 256])
b_conv6 = bias_variable([256])
h_conv6 = tf.nn.relu(conv2d(h_conv5, W_conv6) + b_conv6)

W_conv7 = weight_variable_conv([3, 3, 256, 256])
b_conv7 = bias_variable([256])
h_conv7 = tf.nn.relu(conv2d(h_conv6, W_conv7) + b_conv7)

"""
W_conv8 = weight_variable_conv([3, 3, 256, 256])
b_conv8 = bias_variable([256])
h_conv8 = tf.nn.relu(conv2d(h_conv7, W_conv8) + b_conv8)
#bn8 = batch_normalization(256, h_conv8)
#h_pool8 = max_pool_2x2(bn8)
h_pool8 = max_pool_2x2(h_conv8)
"""
h_pool8 = max_pool_2x2(h_conv7)


W_conv9 = weight_variable_conv([3, 3, 256, 512])
b_conv9 = bias_variable([512])
h_conv9 = tf.nn.relu(conv2d(h_pool8, W_conv9) + b_conv9)

W_conv10 = weight_variable_conv([3, 3, 512, 512])
b_conv10 = bias_variable([512])
h_conv10 = tf.nn.relu(conv2d(h_conv9, W_conv10) + b_conv10)

W_conv11 = weight_variable_conv([3, 3, 512, 512])
b_conv11 = bias_variable([512])
h_conv11 = tf.nn.relu(conv2d(h_conv10, W_conv11) + b_conv11)

"""
W_conv12 = weight_variable_conv([3, 3, 512, 512])
b_conv12 = bias_variable([512])
h_conv12 = tf.nn.relu(conv2d(h_conv11, W_conv12) + b_conv12)
#bn12 = batch_normalization(512, h_conv12)
#h_pool12 = max_pool_2x2(bn12)
h_pool12 = max_pool_2x2(h_conv12)
"""
h_pool12 = max_pool_2x2(h_conv11)

W_conv13 = weight_variable_conv([3, 3, 512, 512])
b_conv13 = bias_variable([512])
h_conv13 = tf.nn.relu(conv2d(h_pool12, W_conv13) + b_conv13)

W_conv14 = weight_variable_conv([3, 3, 512, 512])
b_conv14 = bias_variable([512])
h_conv14 = tf.nn.relu(conv2d(h_conv13, W_conv14) + b_conv14)

W_conv15 = weight_variable_conv([3, 3, 512, 512])
b_conv15 = bias_variable([512])
h_conv15 = tf.nn.relu(conv2d(h_conv14, W_conv15) + b_conv15)

"""
W_conv16 = weight_variable_conv([3, 3, 512, 512])
b_conv16 = bias_variable([512])
h_conv16 = tf.nn.relu(conv2d(h_conv15, W_conv16) + b_conv16)
#bn16 = batch_normalization(512, h_conv16)
#h_pool16 = max_pool_2x2(bn16)
h_pool16 = max_pool_2x2(h_conv16)
"""
h_pool16 = max_pool_2x2(h_conv15)

# 全結合層1
FC_INPUT_NUM = math.ceil(INPUT_W/32) * math.ceil(INPUT_H/32) * 512
W_fc1 = weight_variable([FC_INPUT_NUM, HIDE])
b_fc1 = bias_variable([HIDE])
# 整形
h_pool16 = tf.reshape(h_pool16, [-1, FC_INPUT_NUM])
h_fc1 = tf.nn.relu(tf.matmul(h_pool16, W_fc1) + b_fc1)

# Dropout
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# 全結合層2
W_fc2 = weight_variable([HIDE, HIDE])
b_fc2 = bias_variable([HIDE])
h_fc2 = tf.nn.relu(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

#Dropout
h_fc2_drop = tf.nn.dropout(h_fc2, keep_prob)

# 全結合層3
W_fc3 = weight_variable([HIDE, 1000])
b_fc3 = bias_variable([1000])
h_fc3 = tf.nn.relu(tf.matmul(h_fc2_drop, W_fc3) + b_fc3)

# 出力層
W_fc4 = weight_variable([1000, CATEGORY_NUM])
b_fc4 = bias_variable([CATEGORY_NUM])
y = tf.nn.softmax(tf.matmul(h_fc3, W_fc4) + b_fc4)

# 学習モデルを作成(損失とオプティマイザーを定義)
y_ = tf.placeholder(tf.float32, [None, CATEGORY_NUM])
cross_entropy = -tf.reduce_sum(y_ * tf.log(y))

loss = cross_entropy / BATCH_SIZE
train_step = tf.train.AdamOptimizer(0.00001).minimize(loss)

saver = tf.train.Saver()
saver.restore(sess, "./model/model.ckpt")

# 学習データ
train_x = np.zeros([TRAIN_NUM, INPUT_NUM])
train_t = np.zeros([TRAIN_NUM, CATEGORY_NUM])

# 読み込み、前処理
def img_flat_read(fpass):
    img = cv2.imread(fpass)

    # 28x28にリサイズ
    img_28 = cv2.resize(img, (INPUT_W, INPUT_H))
    # 1列にし0-1のfloatに
    img_flat = img_28.reshape([1, INPUT_NUM]) / 255.0

    # 正規化
    ave = sum(img_flat[0])/len(img_flat[0])
    #std = math.sqrt(np.var(img_flat[0]))
    #img_normal = (img_flat-ave)
    return img_flat, img

# ココア
for i in range(TRAIN_ONE_NUM):
    train_x[i],_ = img_flat_read('train/cocoa/'+str(i)+'.jpg')
    train_t[i][0] = 1
    
# チノ
for i in range(TRAIN_ONE_NUM):
    train_x[i+TRAIN_ONE_NUM],_ = img_flat_read('train/chino/'+str(i)+'.jpg')
    train_t[i+TRAIN_ONE_NUM][1] = 1
    
# リゼ
for i in range(TRAIN_ONE_NUM):
    train_x[i+TRAIN_ONE_NUM*2],_ = img_flat_read('train/rize/'+str(i)+'.jpg')
    train_t[i+TRAIN_ONE_NUM*2][2] = 1

# 千夜
for i in range(TRAIN_ONE_NUM):
    train_x[i+TRAIN_ONE_NUM*3],_ = img_flat_read('train/chiya/'+str(i)+'.jpg')
    train_t[i+TRAIN_ONE_NUM*3][3] = 1

# シャロ
for i in range(TRAIN_ONE_NUM):
    train_x[i+TRAIN_ONE_NUM*4],_ = img_flat_read('train/syaro/'+str(i)+'.jpg')
    train_t[i+TRAIN_ONE_NUM*4][4] = 1



# 学習
for i in range(0):
    
    train_step.run(feed_dict={x:train_x, y_:train_t, keep_prob: 0.5})
    if i % 10 == 0:
        loss_val=sess.run(loss, feed_dict={x:train_x, y_:train_t, keep_prob: 1.0})
        print("Step:%d, Loss:%f"%(i,loss_val))

#saver.save(sess, "./model/model.ckpt")

# 結果
TEST_NUM = 16

test_img = np.array([(cv2.imread('test/'+str(i)+'.jpg')) for i in range(TEST_NUM)])
test_x = np.zeros([TEST_NUM, INPUT_NUM])

for i in range(TEST_NUM):
    img_flat, _ = img_flat_read('test/'+str(i)+'.jpg')
    test_x[i] = img_flat
    
# シャッフル
#np.random.shuffle(test_x)

out = sess.run(y, feed_dict={x:test_x, keep_prob: 1.0})

for i in range(len(out)):
    index = np.argmax(out[i])
    if index == 0:
        print("ココア " + str(int(out[i][0]*100.0)) + "%")
    elif index == 1:
        print("チノ " + str(int(out[i][1]*100.0)) + "%")
    elif index == 2:
        print("リゼ " + str(int(out[i][2]*100.0)) + "%")
    elif index == 3:
        print("千夜 " + str(int(out[i][3]*100.0)) + "%")
    elif index == 4:
        print("シャロ " + str(int(out[i][4]*100.0)) + "%")
    #img = test_x[i].reshape([INPUT_W, INPUT_H, INPUT_C])
    #img = cv2.resize(img, (128, 128))
    test_img[i] = cv2.resize(test_img[i], (256, 256))
    cv2.imshow('image', test_img[i])
    cv2.waitKey(0)

cv2.destroyAllWindows()
