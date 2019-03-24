# /*
#  * @Author: thunder_dz 
#  * @Date: 2019-03-24 10:24:03 
#  * @Last Modified by:   thunder_dz 
#  * @Last Modified time: 2019-03-24 10:24:03 
#  */

import tensorflow as tf

global_step = tf.Variable(0,trainable=False)

init_learning_rate = 0.1

learning_rate = tf.train.exponential_decay(init_learning_rate,global_step=global_step,decay_steps=10,decay_rate=0.9)

opt = tf.train.GradientDescentOptimizer(learning_rate)

#定义一个op,让global_step+1完成计步
add_global = global_step.assign_add(1)

with tf.Session() as sess:
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    print(sess.run(learning_rate))
    for i in range(20):
        g,rate = sess.run([add_global, learning_rate])
        #global_step=global_step+1
        #g,rate = sess.run([global_step, learning_rate])
        print(g, rate)
