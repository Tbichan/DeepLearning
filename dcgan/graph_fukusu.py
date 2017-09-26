# coding: utf-8

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

class NN:
    val = 0
    
    def __init__(self):
        

        # モデル作成
        H1 = 50
        H2 = 30
        self.x = tf.placeholder(tf.float32, [None, 1])
        self.W1 = tf.Variable(tf.random_normal([1, H1], mean=0.0, stddev=0.05))
        self.b1 = tf.Variable(tf.zeros([H1]))
        self.y1 = tf.nn.relu(tf.matmul(self.x, self.W1) + self.b1)

        self.W2 = tf.Variable(tf.random_normal([H1, H2], mean=0.0, stddev=0.05))
        self.b2 = tf.Variable(tf.zeros([H2]))
        self.y2 = tf.nn.relu(tf.matmul(self.y1, self.W2) + self.b2)

        self.W3 = tf.Variable(tf.random_normal([H2, 1], mean=0.0, stddev=0.05))
        self.b3 = tf.Variable(tf.zeros([1]))
        self.y  = tf.matmul(self.y2, self.W3) + self.b3

        # 学習モデル作成(損失とオプティマイザーを定義)
        self.t = tf.placeholder(tf.float32,[None, 1])
        self.loss = 0.5 * tf.reduce_sum(tf.square(self.y-self.t))
        self.train_step = tf.train.AdamOptimizer(0.01).minimize(self.loss)

        
        
    def out(self):
        print(self.val)
        
    def train(self, train_x, train_t):
        self.train_step.run(feed_dict={self.x:train_x, self.t:train_t})
        
    def loss_val(self, train_x, train_t):
        return sess.run(self.loss, feed_dict={self.x:train_x, self.t:train_t})
        
    def test(self, test_x):
        return sess.run(self.y, feed_dict={self.x:test_x})
        
nn = NN()
nn2 = NN()

# セッション作成
sess = tf.InteractiveSession()

# 初期化
tf.global_variables_initializer().run()


# 学習データ
train_x = np.array([i / 100 * 2 * np.pi for i in range(100)])
train_x = train_x.reshape([100,1])

# 教師データ
train_t = np.array([np.sin(i / 100 * 2.0 * np.pi) for i in range(100)])
train_t = train_t.reshape([100,1])

train_t2 = np.array([np.cos(i / 100 * 2.0 * np.pi) for i in range(100)])
train_t2 = train_t2.reshape([100,1])

# 学習
for i in range(10000):
    nn.train(train_x, train_t)
    nn2.train(train_x, train_t2)
    #nn.train_step.run(feed_dict={x:train_x, t:train_t})
    if i % 100 == 0:
        print(nn.loss_val(train_x, train_t))
        #loss_val = sess.run(loss, feed_dict={x:train_x, t:train_t})

# 結果
# 学習データ
test_x = np.array([i / 201 * 2.0 * np.pi for i in range(201)])
test_x = test_x.reshape([201,1])

test_t = np.array([np.sin(i / 201 * 2.0 * np.pi) for i in range(201)])

print("結果")
test_out = nn.test(test_x)
test_out2 = nn2.test(test_x)
#print(test_out)

plt.plot(test_x, test_out)
plt.plot(test_x, test_out2)
#plt.plot(test_x, test_t)
plt.show()
