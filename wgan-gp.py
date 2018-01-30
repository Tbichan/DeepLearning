
# coding: utf-8

# In[1]:

import os
os.environ["CHAINER_TYPE_CHECK"] = "0"

import math
import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_mldata
import chainer
from chainer import cuda, Chain, optimizers, Variable, serializers, Link
import chainer.functions as F
import chainer.links as L
import cupy
import cv2


# In[2]:

xp = cuda.cupy
#print(chainer.functions.Linear(1,1).type_check_enable)


# In[3]:

class Generator(chainer.Chain):
    
    def __init__(self, z_dim):
        initializer = chainer.initializers.Normal(scale=0.02)
        super(Generator, self).__init__(
            l1 = L.Linear(z_dim, 8*8*512, initialW=initializer),
            
            #dc0 = L.Deconvolution2D(2048, 1024, 4, stride=2, pad=1, initialW=initializer),
            dc1 = L.Deconvolution2D(512, 256, 4, stride=2, pad=1, initialW=initializer),
            dc2 = L.Deconvolution2D(256, 128, 4, stride=2, pad=1, initialW=initializer),
            dc3 = L.Deconvolution2D(128, 64, 4, stride=2, pad=1, initialW=initializer),
            dc4 = L.Deconvolution2D(64, 32, 4, stride=2, pad=1, initialW=initializer),
            dc5 = L.Deconvolution2D(32, 3, 4, stride=2, pad=1, initialW=initializer),
            
            #c1 = L.Convolution2D(64, 3, 5, stride=1, pad=2, initialW=initializer),
            
            bn0 = L.BatchNormalization(8*8*512),
            bn1 = L.BatchNormalization(256),
            bn2 = L.BatchNormalization(128),
            bn3 = L.BatchNormalization(64),
            bn4 = L.BatchNormalization(32),
        )
    
    def __call__(self, z):
        h = F.leaky_relu(self.bn0(self.l1(z)))
        
        h = F.reshape(h, (z.data.shape[0], 512, 8, 8))
        
        #h = F.relu(self.bn1(self.dc0(h)))
        h = F.leaky_relu(self.bn1(self.dc1(h)))
        h = F.leaky_relu(self.bn2(self.dc2(h)))
        h = F.leaky_relu(self.bn3(self.dc3(h)))
        h = F.leaky_relu(self.bn4(self.dc4(h)))
        return self.dc5(h)
        


# In[4]:

class MinibatchStddev(Link):
    def __init__(self, ch):
        super(MinibatchStddev, self).__init__()
        
        self.eps = 1.0

    def __call__(self, x):
        mean = F.mean(x, axis=0, keepdims=True)
        dev = x - F.broadcast_to(mean, x.shape)
        devdev = dev * dev
        var = F.mean(devdev, axis=0, keepdims=True) # using variance instead of stddev
        stddev_mean = F.mean(var)
        new_channel = F.broadcast_to(stddev_mean, (x.shape[0], 1, x.shape[2], x.shape[3]))
        h = F.concat((x, new_channel), axis=1)
        return h

class Critic(Chain):
    
    def __init__(self):
        initializer = chainer.initializers.Normal(scale=0.02)
        super(Critic, self).__init__(
            
            c0 = L.Convolution2D(3, 32, 4, stride=2, pad=1, initialW=initializer),
            c1 = L.Convolution2D(32, 64, 4, stride=2, pad=1, initialW=initializer),
            c2 = L.Convolution2D(64, 128, 4, stride=2, pad=1, initialW=initializer),
            c3 = L.Convolution2D(128, 256, 4, stride=2, pad=1, initialW=initializer),
            
            # minibatch正規化
            stddev = MinibatchStddev(256),
            
            c4 = L.Convolution2D(256 + 1, 512, 4, stride=2, pad=1, initialW=initializer),
        
            l1 = L.Linear(8*8*512, 1, initialW=initializer)
        
        )
        
    #def clamp(self, lower=-0.01, upper=0.01):
    #    for params in self.params():
    #        params_clipped = F.clip(params, lower, upper)
    #        params.data = params_clipped.data
    
    def __call__(self, x):
        h = F.leaky_relu(self.c0(x))
        h = F.leaky_relu(self.c1(h))
        h = F.leaky_relu(self.c2(h))
        h = F.leaky_relu(self.c3(h))
        h = self.stddev(h)
        h = F.leaky_relu(self.c4(h))
        
        return self.l1(h)


# In[5]:

init=100500

gen = Generator(100)
if init != 0:
    serializers.load_npz("generator.npz", gen)

gpu = 0
chainer.cuda.get_device(gpu).use()
gen.to_gpu(gpu) # GPUを使うための処理

gen_optimizer = optimizers.Adam(alpha=0.0001, beta1=0, beta2=0.99)
#gen_optimizer = optimizers.RMSprop(lr=0.00005)
gen_optimizer.setup(gen)



# Critic
cri = Critic()
if init != 0:
    serializers.load_npz("critic.npz", cri)
cri.to_gpu(gpu) # GPUを使うための処理

cri_optimizer = optimizers.Adam(alpha=0.0001, beta1=0, beta2=0.99) # 0.001
#cri_optimizer = optimizers.RMSprop(lr=0.00005)
cri_optimizer.setup(cri)
#cri_optimizer.add_hook(chainer.optimizer.WeightDecay(0.0001)) # 0.005


# In[6]:

INPUT_W = 256
INPUT_H = 256
INPUT_C = 3
INPUT_NUM = INPUT_W*INPUT_H*INPUT_C
#TRAIN_NUM = 288*2

# 読み込み、前処理
def img_flat_read(fpass):
    img = cv2.imread(fpass)

    # INPUT_WxINPUT_Hにリサイズ
    img_64 = cv2.resize(img, (INPUT_W, INPUT_H))
    
    # 1列にし-1～1のfloatに
    img_flat = 2.0 * img_64 / 255.0 - 1.0
    img_flat_rev = 2.0 * cv2.flip(img_64, 1) / 255.0 - 1.0
    img_flat = np.transpose(img_flat, (2, 1, 0))
    img_flat_rev = np.transpose(img_flat_rev, (2, 1, 0))
        
    return img_flat, img_flat_rev

class CVBatch:
    def __init__(self, num):
        self.num = num
        self.images = np.zeros([num, INPUT_C, INPUT_W, INPUT_H])
        self.nowCnt = 0
        self.nowIndex = 0
        self.gpu = False
    def loadImage(self, fpass):
        
        self.images[self.nowCnt], self.images[self.nowCnt+1] = img_flat_read(fpass)
        self.nowCnt+=2
    
    def shuffle(self):
        np.random.shuffle(self.images)
    
    def nextBatch(self, n):

        if self.nowIndex+n < self.nowCnt:
            res = self.images[self.nowIndex:self.nowIndex+n]
            self.nowIndex += n
        else:
            if self.gpu:
                xp.random.shuffle(self.images)
            else:
                np.random.shuffle(self.images)
                
            res = self.images[0:n]
            self.nowIndex = n
            print("train_shuffle...")
        #np.random.shuffle(self.images)
        return res
    
    def to_gpu(self):
        self.gpu = True
        self.images = xp.array(self.images, dtype=np.float32)
    

fileList = os.listdir("train/")

# 学習データ読み込み

cvBatch = CVBatch(len(fileList) * 2)


print("TRAIN_TOTAL:" + str(len(fileList) * 2))

i = 0
for file in sorted(fileList):
    print(file)
    #train_x[i],img = img_flat_read('train/' + file)
    CVBatch.loadImage(cvBatch, 'train/' + file)
    i+=1
    
#CVBatch.to_gpu(cvBatch)


# In[ ]:

# シード値設定
np.random.seed(seed=20180101)


# In[ ]:

batch_size = 64
n_critic = 1

zTest = np.random.uniform(-1.0, 1.0, size=[batch_size, 100]).astype(np.float32)
zTest = Variable(xp.array(zTest, dtype=np.float32))

for i in range(1000000):
    
    # critic更新
    for _ in range(n_critic):
        
        z = xp.random.uniform(-1.0, 1.0, size=[batch_size, 100]).astype(np.float32)
    
        # Chainer変数に変換
        z = Variable(z)
        
        # 偽物用
        x1 = gen(z)
        #print(x1.data.shape)
        y1 = cri(x1)

        # 本物用
        x2 = CVBatch.nextBatch(cvBatch, batch_size)
        #x2 = Variable(x2)
        x2 = Variable(xp.array(x2, dtype=np.float32))
        y2 = cri(x2)
        
        # grad
        e = xp.random.uniform(0., 1., (batch_size, 1, 1, 1))
        x_hat = e * x1 + (1 - e) * x2
        grad, = chainer.grad([cri(x_hat)], [x_hat], enable_double_backprop=True)
        grad = F.sqrt(F.batch_l2_norm_squared(grad))
        
        loss_grad = 10 * F.mean_squared_error(grad, xp.ones_like(grad.data))
    
        loss_dr = 0.0001 * y2 * y2
        L_cri = F.mean(-y2 + y1 + loss_dr) + loss_grad
        
        # 重みクリップ
        #cri.clamp()
    
        cri.zerograds()
        L_cri.backward()
        cri_optimizer.update()
    
    # 偽物用
    z = np.random.uniform(-1.0, 1.0, size=[batch_size, 100]).astype(np.float32)
    # Chainer変数に変換
    z = Variable(xp.array(z, dtype=np.float32))
        
    x1 = gen(z)
    y1 = cri(x1)

    L_gen = -F.sum(y1) / (batch_size)
    
    gen.zerograds()
    L_gen.backward()
    gen_optimizer.update()
    
    if i % 10 == 0:

        # 画像出力
        # テストモードに
        with chainer.using_config('train', False), chainer.no_backprop_mode():
            
            # 偽物用
            x_test = gen(zTest)
            y_test = cri(x_test)
            L_gen = -F.mean(y_test)
            """
            # grad
            e = xp.random.uniform(0., 1., (batch_size, 1, 1, 1))
            x_hat = e * x_test + (1 - e) * x2
            grad, = chainer.grad([cri(x_hat)], [x_hat], enable_double_backprop=True)
            grad = F.sqrt(F.batch_l2_norm_squared(grad))

            loss_grad = 10 * F.mean_squared_error(grad, xp.ones_like(grad.data))
            """
            loss_dr = 0.0001 * y2 * y2
            L_cri = F.mean(-y2 + y1 + loss_dr)
            
            g_loss = cuda.to_cpu(L_gen.data)
            c_loss = cuda.to_cpu(L_cri.data)
            
            # w距離
            w_dis = F.mean(-y2 + y1)
            w_dis = cuda.to_cpu(w_dis.data)
            
            print(str(init+i) + ":gen:" + str(g_loss) + ", cri:" + str(c_loss) + ", w_dis:" + str(w_dis))
            
        imgs = cuda.to_cpu(x_test.data)
        imgs = np.transpose(imgs, (0, 3, 2, 1))

        # 画像を連結
        img2 = []

        for k in range(8):
            img2.append(imgs[k*8])

            for j in range(8):
                if j != 0:
                    img2[k] = cv2.hconcat([img2[k], imgs[k*8+j]])
            if k == 0:
                img3 = img2[0]
            else:
                img3 = cv2.vconcat([img3, img2[k]])

        cv2.imwrite('./res/'+str(init+i)+'.jpg', 255 * 0.5 * (img3+1.0))
        
    if i % 500 == 0 and i != 0:
        print("save")
        gen.to_cpu()
        serializers.save_npz("generator.npz", gen)
        gen.to_gpu(gpu)
        cri.to_cpu()
        serializers.save_npz("critic.npz", cri)
        cri.to_gpu(gpu)
        print("ok")
    
    if (init+i) % 10000 == 0 and i != 0:
        print("save")
        gen.to_cpu()
        serializers.save_npz("./weight/generator_" + str(init+i) + ".npz", gen)
        gen.to_gpu(gpu)
        cri.to_cpu()
        serializers.save_npz("./weight/critic_" + str(init+i) + ".npz", cri)
        cri.to_gpu(gpu)
        print("ok")
    


# In[ ]:

gen.to_cpu()
serializers.save_npz("generator.npz", gen)
cri.to_cpu()
serializers.save_npz("critic.npz", cri)


# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:



