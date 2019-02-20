# /*
#  * @Author: thunder_dz 
#  * @Date: 2019-02-19 20:27:56 
#  * @Last Modified by:   thunder_dz 
#  * @Last Modified time: 2019-02-19 20:27:56 
#  */


# global_step在滑动平均、优化器、指数衰减学习率等方面都有用到。
# 这个变量的实际意义非常好理解：代表全局步数，比如在多少步该进行什么操作，现在神经网络训练到多少轮等等，类似于一个钟表。
import tensorflow as tf
import numpy as np
 
x = tf.placeholder(tf.float32, shape=[None, 1], name='x')
y = tf.placeholder(tf.float32, shape=[None, 1], name='y')
w = tf.Variable(tf.constant(0.0))
 
global_steps = tf.Variable(0, trainable=False)
 
 
learning_rate = tf.train.exponential_decay(0.1, global_steps, 10, 0.96, staircase=False)
loss = tf.pow(w*x-y, 2)
 
train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss,global_step=global_steps)

#经过多次修改代码验证之后得出，如果把
#train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss,global_step=global_steps)
#后面部分的global_step=global_steps去掉，global_step的自动加一就会失效，
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(10):
        sess.run(train_step, feed_dict={x:np.linspace(1,2,10).reshape([10,1]),
            y:np.linspace(1,2,10).reshape([10,1])})
        print(sess.run(learning_rate))
        print(sess.run(global_steps))