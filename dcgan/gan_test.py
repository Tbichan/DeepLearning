import numpy as np
import cv2
import tensorflow as tf
import os
import time
import math

# Batch Normalization
def batch_normalization(shape, input, name="bn", withGamma=False):
    with tf.variable_scope(name):
        gamma_init= tf.truncated_normal_initializer(mean=0.0, stddev=0.05)
        gamma = tf.get_variable('gamma', [shape], initializer=gamma_init)
        beta_init= tf.truncated_normal_initializer(mean=0.0, stddev=0.05)
        beta = tf.get_variable('beta', [shape], initializer=beta_init)
        mean, variance = tf.nn.moments(input, [0])
    if withGamma == False:
        return gamma * (input - mean) / tf.sqrt(variance + 1e-5) + beta
    else:
        return gamma * (input - mean) / tf.sqrt(variance + 1e-5) + beta, gamma, beta

def conv2d(input_, output_dim, 
       k_h=5, k_w=5, d_h=2, d_w=2, stddev=0.02,
       name="conv2d", with_w=False):
  with tf.variable_scope(name):
    w = tf.get_variable('W', [k_h, k_w, input_.get_shape()[-1], output_dim],
              initializer=tf.truncated_normal_initializer(stddev=stddev))
    conv = tf.nn.conv2d(input_, w, strides=[1, d_h, d_w, 1], padding='SAME')

    biases = tf.get_variable('b', [output_dim], initializer=tf.constant_initializer(0.0))
    conv = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape())

    if with_w:
        return conv, w, biases
    else:
        return conv


def deconv2d(input_, output_shape,
       k_h=5, k_w=5, d_h=2, d_w=2, stddev=0.02,
       name="deconv2d", with_w=False):
  with tf.variable_scope(name):
    # filter : [height, width, output_channels, in_channels]
    w = tf.get_variable('W', [k_h, k_w, output_shape[-1], input_.get_shape()[-1]],
              initializer=tf.random_normal_initializer(stddev=stddev))
    
    try:
      deconv = tf.nn.conv2d_transpose(input_, w, output_shape=output_shape,
                strides=[1, d_h, d_w, 1])

    # Support for verisons of TensorFlow before 0.7.0
    except AttributeError:
      deconv = tf.nn.deconv2d(input_, w, output_shape=output_shape,
                strides=[1, d_h, d_w, 1])

    biases = tf.get_variable('b', [output_shape[-1]], initializer=tf.constant_initializer(0.0))
    deconv = tf.reshape(tf.nn.bias_add(deconv, biases), deconv.get_shape())

    if with_w:
      return deconv, w, biases
    else:
        return deconv

def lrelu(x, name="lrelu"):
    return tf.maximum(x, x*0.2, name=name)

INPUT_W = 128
INPUT_H = 128
INPUT_C = 3
INPUT_NUM = INPUT_W*INPUT_H*INPUT_C
TRAIN_NUM = 180

# ミニバッチ数
MINI_BATCH = 64

class Generator(object):

    def __init__(self, input, in_num, name='generator'):

        # 重み、バイアスを定義
        #self.input = input
    
        with tf.variable_scope(name):

            # 全結合層
            weight_init= tf.truncated_normal_initializer(mean=0.0, stddev=0.05)
            self.W_fc1 = tf.get_variable('W_fc1', [in_num, 4*4*2048], initializer=weight_init)
            bias_init = tf.constant_initializer(value=0.0)
            self.b_fc1 = tf.get_variable('b_fc1', [4*4*2048], initializer=bias_init)

            linarg = tf.matmul(input, self.W_fc1) + self.b_fc1
            gamma_bn1, beta_bn1 = tf.nn.moments(linarg, [0,1], name='bn1')
            relu_fc1 = tf.nn.relu(tf.nn.batch_normalization(linarg, gamma_bn1, beta_bn1, None , None,1e-5,name='bn1'))
            
            #bn1, gamma_bn1, beta_bn1 = batch_normalization(4*4*2048, linarg, name='bn1', withGamma=True)
            #relu_fc1 = tf.nn.relu(bn1)

            self.tmp = relu_fc1
    
            # 逆畳み込み層1
            relu1_image = tf.reshape(relu_fc1, [MINI_BATCH, 4, 4, 2048])
            deconv2, self.W_deconv2, self.b_deconv2 = \
                     deconv2d(relu1_image, [MINI_BATCH, 8, 8, 1024], name='deconv_2', with_w=True)
            gamma_bn2, beta_bn2 = tf.nn.moments(deconv2, [0,1,2], name='bn2')
            h_conv2 = tf.nn.relu(tf.nn.batch_normalization(deconv2, gamma_bn2, beta_bn2, None , None,1e-5,name='bn2'))

            # 逆畳み込み層2
            deconv3, self.W_deconv3, self.b_deconv3 = \
                     deconv2d(h_conv2, [MINI_BATCH, 16, 16, 512], name='deconv_3', with_w=True)
            # Batch Normalization, ReLu
            gamma_bn3, beta_bn3 = tf.nn.moments(deconv3, [0,1,2], name='bn3')
            h_conv3 = tf.nn.relu(tf.nn.batch_normalization(deconv3, gamma_bn3, beta_bn3, None , None,1e-5,name='bn3'))

            # 逆畳み込み層3
            deconv4, self.W_deconv4, self.b_deconv4 = \
                     deconv2d(h_conv3, [MINI_BATCH, 32, 32, 256], name='deconv_4', with_w=True)
            # Batch Normalization, ReLu
            gamma_bn4, beta_bn4 = tf.nn.moments(deconv4, [0,1,2], name='bn4')
            h_conv4 = tf.nn.relu(tf.nn.batch_normalization(deconv4, gamma_bn4, beta_bn4, None , None,1e-5,name='bn4'))

            # 逆畳み込み層4
            deconv5, self.W_deconv5, self.b_deconv5 = \
                     deconv2d(h_conv4, [MINI_BATCH, 64, 64, 128], name='deconv_5', with_w=True)
            # Batch Normalization, ReLu
            gamma_bn5, beta_bn5 = tf.nn.moments(deconv5, [0,1,2], name='bn5')
            h_conv5 = tf.nn.relu(tf.nn.batch_normalization(deconv5, gamma_bn5, beta_bn5, None , None,1e-5,name='bn5'))

            # 逆畳み込み層5
            deconv6, self.W_deconv6, self.b_deconv6 = \
                     deconv2d(h_conv5, [MINI_BATCH, 128, 128, 3], name='deconv_6', with_w=True)
            
            self.output = tf.nn.tanh(deconv6)
            
    def output(self):
        return self.output

class Discriminator(object):

    def __init__(self, input, reuse=False, name='discriminator'):

        # 重み、バイアスを定義
        self.input = input

        if not reuse:
            with tf.variable_scope(name):
                
                # 畳み込み0
                input_image = tf.reshape(input, [-1, 128, 128, 3])
                weight_init = tf.truncated_normal_initializer(mean=0.0, stddev=0.05)
                bias_init = tf.constant_initializer(value=0.0)
                
                self.W_conv0 = tf.get_variable('W_conv_0', [5, 5, 3, 128], initializer=weight_init)
                self.b_conv0 = tf.get_variable('b_conv_0', [128], initializer=bias_init)
                h_conv0 = lrelu(tf.nn.conv2d(input_image, self.W_conv0, strides=[1,2,2,1], padding='SAME') + self.b_conv0, name='lreru_0')

                # 畳み込み1
                self.W_conv1 = tf.get_variable('W_conv_1', [5, 5, 128, 256], initializer=weight_init)
                self.b_conv1 = tf.get_variable('b_conv_1', [256], initializer=bias_init)
                aff_conv1 = tf.nn.conv2d(h_conv0, self.W_conv1, strides=[1,2,2,1], padding='SAME') + self.b_conv1
                # Batch Normalization
                gamma_bn1, beta_bn1 = tf.nn.moments(aff_conv1, [0,1,2], name='bn1')
                h_conv1 = lrelu(tf.nn.batch_normalization(aff_conv1, gamma_bn1, beta_bn1, None , None,1e-5,name='bn1'),  name='lreru_1')
                #h_conv1 = lrelu(aff_conv1)

                # 畳み込み2
                self.W_conv2 = tf.get_variable('W_conv_2', [5, 5, 256, 512], initializer=weight_init)
                self.b_conv2 = tf.get_variable('b_conv_2', [512], initializer=bias_init)
                aff_conv2 = tf.nn.conv2d(h_conv1, self.W_conv2, strides=[1,2,2,1], padding='SAME') + self.b_conv2
                # Batch Normalization
                gamma_bn2, beta_bn2 = tf.nn.moments(aff_conv2, [0,1,2], name='bn2')
                h_conv2 = lrelu(tf.nn.batch_normalization(aff_conv2, gamma_bn2, beta_bn2, None , None,1e-5,name='bn2'),  name='lreru_2')
                #h_conv2 = lrelu(aff_conv2)

                # 畳み込み3
                self.W_conv3 = tf.get_variable('W_conv_3', [5, 5, 512, 1024], initializer=weight_init)
                self.b_conv3 = tf.get_variable('b_conv_3', [1024], initializer=bias_init)
                aff_conv3 = tf.nn.conv2d(h_conv2, self.W_conv3, strides=[1,2,2,1], padding='SAME') + self.b_conv3
                # Batch Normalization
                gamma_bn3, beta_bn3 = tf.nn.moments(aff_conv3, [0,1,2], name='bn3')
                h_conv3 = lrelu(tf.nn.batch_normalization(aff_conv3, gamma_bn3, beta_bn3, None , None,1e-5,name='bn3'),  name='lreru_3')
                #h_conv3 = lrelu(aff_conv3)
            
                # 畳み込み4
                self.W_conv4 = tf.get_variable('W_conv_4', [5, 5, 1024, 2048], initializer=weight_init)
                self.b_conv4 = tf.get_variable('b_conv_4', [2048], initializer=bias_init)
                aff_conv4 = tf.nn.conv2d(h_conv3, self.W_conv4, strides=[1,2,2,1], padding='SAME') + self.b_conv4
                # Batch Normalization
                gamma_bn4, beta_bn4 = tf.nn.moments(aff_conv4, [0,1,2], name='bn4')
                h_conv4 = lrelu(tf.nn.batch_normalization(aff_conv4, gamma_bn4, beta_bn4, None , None,1e-5,name='bn4'),  name='lreru_4')
                #h_conv4 = lrelu(aff_conv4)
                
                # 全結合層
                self.W_fc5 = tf.get_variable('W_fc5', [4*4*2048, 1], initializer=weight_init)
                self.b_fc5 = tf.get_variable('b_fc5', [1], initializer=bias_init)
                h_conv4_flat = tf.reshape(h_conv4, [-1, 4*4*2048])
                linarg = tf.matmul(h_conv4_flat, self.W_fc5) + self.b_fc5

                # DropOut
                #h5_drop = tf.nn.dropout(h5, keep_prob)

                # 全結合層
                #self.W_fc6 = tf.get_variable('W_fc6', [8, 1], initializer=weight_init)
                #self.b_fc6 = tf.get_variable('b_fc6', [1], initializer=bias_init)
                #linarg = tf.matmul(h5_drop, self.W_fc6) + self.b_fc6
                
                #h5 = tf.nn.sigmoid(linarg)
                
        else:   # 重み共有
            with tf.variable_scope(name, reuse=True):

                # 畳み込み0
                input_image = tf.reshape(input, [-1, 128, 128, 3])
                
                self.W_conv0 = tf.get_variable('W_conv_0', [5, 5, 3, 128])
                self.b_conv0 = tf.get_variable('b_conv_0', [128])
                h_conv0 = lrelu(tf.nn.conv2d(input_image, self.W_conv0, strides=[1,2,2,1], padding='SAME') + self.b_conv0, name='lreru_0')

                # 畳み込み1
                self.W_conv1 = tf.get_variable('W_conv_1', [5, 5, 128, 256])
                self.b_conv1 = tf.get_variable('b_conv_1', [256])
                aff_conv1 = tf.nn.conv2d(h_conv0, self.W_conv1, strides=[1,2,2,1], padding='SAME') + self.b_conv1
                # Batch Normalization
                gamma_bn1, beta_bn1 = tf.nn.moments(aff_conv1, [0,1,2], name='bn1')
                h_conv1 = lrelu(tf.nn.batch_normalization(aff_conv1, gamma_bn1, beta_bn1, None , None,1e-5,name='bn1'),  name='lreru_1')
                #h_conv1 = lrelu(aff_conv1)
                
                # 畳み込み2
                self.W_conv2 = tf.get_variable('W_conv_2', [5, 5, 256, 512])
                self.b_conv2 = tf.get_variable('b_conv_2', [512])
                aff_conv2 = tf.nn.conv2d(h_conv1, self.W_conv2, strides=[1,2,2,1], padding='SAME') + self.b_conv2
                # Batch Normalization
                gamma_bn2, beta_bn2 = tf.nn.moments(aff_conv2, [0,1,2], name='bn2')
                h_conv2 = lrelu(tf.nn.batch_normalization(aff_conv2, gamma_bn2, beta_bn2, None , None,1e-5,name='bn2'),  name='lreru_2')
                #h_conv2 = lrelu(aff_conv2)
                
                # 畳み込み3
                self.W_conv3 = tf.get_variable('W_conv_3', [5, 5, 512, 1024])
                self.b_conv3 = tf.get_variable('b_conv_3', [1024])
                aff_conv3 = tf.nn.conv2d(h_conv2, self.W_conv3, strides=[1,2,2,1], padding='SAME') + self.b_conv3
                # Batch Normalization
                gamma_bn3, beta_bn3 = tf.nn.moments(aff_conv3, [0,1,2], name='bn3')
                h_conv3 = lrelu(tf.nn.batch_normalization(aff_conv3, gamma_bn3, beta_bn3, None , None,1e-5,name='bn3'),  name='lreru_3')
                #h_conv3 = lrelu(aff_conv3)

                # 畳み込み4
                self.W_conv4 = tf.get_variable('W_conv_4', [5, 5, 1024, 2048])
                self.b_conv4 = tf.get_variable('b_conv_4', [2048])
                aff_conv4 = tf.nn.conv2d(h_conv3, self.W_conv4, strides=[1,2,2,1], padding='SAME') + self.b_conv4
                # Batch Normalization
                gamma_bn4, beta_bn4 = tf.nn.moments(aff_conv4, [0,1,2], name='bn4')
                h_conv4 = lrelu(tf.nn.batch_normalization(aff_conv4, gamma_bn4, beta_bn4, None , None,1e-5,name='bn4'),  name='lreru_4')
                #h_conv4 = lrelu(aff_conv4)

                # 全結合層
                self.W_fc5 = tf.get_variable('W_fc5', [4*4*2048, 1])
                self.b_fc5 = tf.get_variable('b_fc5', [1])
                h_conv4_flat = tf.reshape(h_conv4, [-1, 4*4*2048])
                linarg = tf.matmul(h_conv4_flat, self.W_fc5) + self.b_fc5

                # DropOut
                #h5_drop = tf.nn.dropout(h5, keep_prob)

                # 全結合層
                #self.W_fc6 = tf.get_variable('W_fc6', [8, 1], initializer=weight_init)
                #self.b_fc6 = tf.get_variable('b_fc6', [1], initializer=bias_init)
                #linarg = tf.matmul(h5_drop, self.W_fc6) + self.b_fc6
                
                #h5 = tf.nn.sigmoid(linarg)
                
        self.output = linarg
        self.output_aff = linarg

    def output(self):
        return self.output, self.output_aff



z = tf.placeholder("float", [None, 100])
images = tf.placeholder("float", [None, 128*128*3])
keep_prob = tf.placeholder(tf.float32)

def make_model(z, images):
    g = Generator(z, 100)
    fake_img = Generator.output(g)
    d_fake = Discriminator(fake_img)                # 偽物用
    d_fake_out, d_logits_f = Discriminator.output(d_fake)
    d_true = Discriminator(images, reuse=True)      # 本物用(重み共有)
    d_true_out, d_logits_t = Discriminator.output(d_true)

    # 損失関数
    

    
    """
    zeros = tf.random_uniform([MINI_BATCH, 1], minval=0.0, maxval=0.3)
    ones = tf.random_uniform([MINI_BATCH, 1], minval=0.7, maxval=1.0)

    tf.add_to_collection('d_losses', tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_t, labels=ones)))
    tf.add_to_collection('d_losses', tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_f, labels=zeros)))
    
    tf.add_to_collection('g_losses', tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_f, labels=ones)))
    #tf.add_to_collection('g_losses', tf.reduce_mean(
    #    tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_t, labels=zeros)))

    
    tf.add_to_collection('d_losses', tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_t, labels= tf.ones_like(d_true_out))))
    tf.add_to_collection('d_losses', tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_f, labels= zeros)))
    tf.add_to_collection('g_losses', tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_f, labels= tf.ones_like(d_fake_out))))

    
    tf.add_to_collection('g_losses', tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_t, labels= tf.zeros_like(d_true_out))))
    """

    #g_loss = tf.add_n(tf.get_collection('g_losses'), name='total_g_loss')
    #d_loss = tf.add_n(tf.get_collection('d_losses'), name='total_d_loss')

    g_loss = -tf.reduce_mean(d_logits_f)
    d_loss = tf.reduce_mean(-d_logits_t) + tf.reduce_mean(d_logits_f)

    # 変数初期化
    t_vars = tf.trainable_variables()

    d_vars = [var for var in t_vars if 'discriminator' in var.name]
    g_vars = [var for var in t_vars if 'generator' in var.name]

    return d_loss, g_loss, d_vars, g_vars, fake_img, d_fake_out, d_true_out


# モデル作成
d_loss, g_loss, d_vars, g_vars, fake_img, d_fake, d_true = make_model(z, images)

# オプティマイザ定義
#learning_g_rate = 0.0002
#learning_d_rate = 0.0002
learning_g_rate = 0.00005
learning_d_rate = 0.00005

d_optim = tf.train.RMSPropOptimizer(learning_d_rate) \
              .minimize(d_loss, var_list=d_vars)

g_optim = tf.train.RMSPropOptimizer(learning_g_rate) \
.minimize(g_loss, var_list=g_vars)

w_clip = [var.assign(tf.clip_by_value(var, -0.01, 0.01)) for var in d_vars]

init = tf.global_variables_initializer()

# 読み込み、前処理
def img_flat_read(fpass):
    img = cv2.imread(fpass)

    # INPUT_WxINPUT_Hにリサイズ
    img_64 = cv2.resize(img, (INPUT_W, INPUT_H))
    # 1列にし-1～1のfloatに
    img_flat = 2.0 * img_64.reshape([1, INPUT_NUM]) / 255.0 - 1.0
    img_flat_rev = 2.0 * cv2.flip(img_64, 1).reshape([1, INPUT_NUM]) / 255.0 - 1.0
        
    return img_flat, img_flat_rev

# 学習データ
#train_x = np.zeros([TRAIN_NUM, INPUT_NUM])
#train_t = np.zeros([TRAIN_NUM, CATEGORY_NUM])

class CVBatch:
    def __init__(self, num):
        self.num = num
        self.images = np.zeros([TRAIN_NUM, INPUT_NUM])
        self.nowCnt = 0
        self.nowIndex = 0
    def loadImage(self, fpass):
        self.images[self.nowCnt], img = img_flat_read(fpass)
        self.nowCnt+=1
        return img
    
    def nextBatch(self, n):
        start = self.nowIndex
        end = self.nowIndex+n
        if end >= self.num:
            end = self.num - 1
            self.nowIndex = 0
            res = self.images[start:end]

            # シャッフル
            #l = len(self.images)
            #rnd1 = np.random.randint(0, l)
            #np.random.shuffle(self.images)
        else:
            self.nowIndex = end
            res = self.images[start:end]
        return res, end - start
        
    

#fileList = os.listdir("train/")

# 学習データ読み込み

#cvBatch = CVBatch(TRAIN_NUM)

#i = 0
#for file in sorted(fileList):
    #print(file)
    #train_x[i],img = img_flat_read('train/' + file)
    #img = CVBatch.loadImage(cvBatch, 'train/' + file)
#    i+=1



# 学習
with tf.Session() as sess:
    sess.run(init)

    saver = tf.train.Saver()

    saver.restore(sess, "./modelTest/model.ckpt")

    zInput = np.random.uniform(-1.0, 1.0, size=[MINI_BATCH, 100]).astype(np.float32)

    for i in range(10000):

        # バッチをとってくる
        #img_input, num = CVBatch.nextBatch(cvBatch, MINI_BATCH)
        zInput+=np.random.uniform(-0.075, 0.075, size=[MINI_BATCH, 100]).astype(np.float32)
        zInput = zInput.clip(min=-1.0,max=1.0)
        #zInput = np.zeros([MINI_BATCH, 100])
        #for k in range(MINI_BATCH):
            #zInput[k][k] = math.sin(i * 2 * 3.141592 / 1000)
        #zInput[0][13] = -0.7
        #zInput[0][63] = 0.8
        
        #zInput = np.random.uniform(-1.0, 1.0, size=[MINI_BATCH, 100]).astype(np.float32)
        
        #print(zInput)
        
        img = sess.run(fake_img, feed_dict={z:zInput, keep_prob: 0.5})

        if i % 1 == 0:

            #cv2.imwrite('./create/'+str(i)+'.jpg', 255 * 0.5 * (img[0]+1.0))

            
            # 画像を連結
            #img2 = [img[0],img[8],img[12],img[18],img[24],img[30]]
            img2 = []

            size = 8
            for k in range(size):
                for j in range(size):
                    if j != 0:
                        img2[k] = cv2.hconcat([img2[k], img[k*size+j]])
                    else:
                        img2.append(img[k*size])
                if k == 0:
                    img3 = img2[0]
                else:
                    img3 = cv2.vconcat([img3, img2[k]])
            img4 = (img3) * 255    #img4 = (img3+1.0) * 127
            #cv2.imshow('image', img3)
            #cv2.waitKey(0)
                
            cv2.imwrite('./create/'+str(i)+'.jpg', img4)
            print(i)
            
    
