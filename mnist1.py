# /*
#  * @Author: thunder_dz 
#  * @Date: 2019-02-18 17:44:08 
#  * @Last Modified by: thunder_dz
#  * @Last Modified time: 2019-02-18 19:42:19
#  */

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

input_node = 784    #输入层节点数，对于mnist数据集等于图片像素
output_node = 10    #输出层节点数，等于类别数目

layer_node = 500    #隐藏层节点数

batch_size = 100    #一个训练batch中的训练数据个数。数字越小时，越接近随机梯度下降，数字越大，越接近梯度下降。

learning_rate_base = 0.8
learning_rate_decay = 0.99    #学习率的衰减率

redularizetion_rate = 0.0001  #描述模型复杂的的正则化项在损失函数中的系数
training_steps = 30000
moving_average_decay = 0.99   #滑动平均衰减率,越大模型越稳定

#前向传播
def inference(input_tensor, avg_class, weights1, biases1, weights2, biases2):
    
    if avg_class == None:     #没有滑动平均类时，直接使用参数当前的取值
        layer1 = tf.nn.relu(tf.matmul(input_tensor, weights1) + biases1)
        return tf.matmul(layer1,weights2) + biases2
    else:
        layer1 = tf.nn.relu(tf.matmul(input_tensor,avg_class.average(weights1)) + 
                            avg_class.average(biases1))
        return tf.matmul(layer1, avg_class.average(weights2)) + avg_class.average(biases2)


def train(mnist):
    x = tf.placeholder(tf.float32, [None, input_node], name="x-input")
    y_ = tf.placeholder(tf.float32, [None, output_node], name="y_input")

    #tf.truncated_normal(),从截断的正态分布中输出随机值。 
    #生成的值服从具有指定平均值和标准偏差的正态分布，如果生成的值大于平均值2个标准偏差的值则丢弃重新选择。
    weights1 = tf.Variable(tf.truncated_normal([input_node, layer_node],stddev=0.1))
    biases1 = tf.Variable(tf.constant(0.1, shape=[layer_node]))             #维度为layer_node的行向量

    weights2 = tf.Variable(tf.truncated_normal([layer_node, output_node],stddev=0.1))
    biases2 = tf.Variable(tf.constant(0.1, shape=[output_node]))

    #计算当前参数在神经网络中的前向传播结果

    y = inference(x, None, weights1, biases1, weights2, biases2)

    #滑动平均操作
    global_step = tf.Variable(0, trainable=False)       #trainable=False 该变量不可被训练

    variable_averages = tf.train.ExponentialMovingAverage(moving_average_decay, global_step)

    variables_average_op = variable_averages.apply(tf.trainable_variables())
    
    average_y = inference(x, variable_averages, weights1, biases1, weights2, biases2)


    #当问题只有一个正确答案时，可以用tf.nn.sparse_softmax_cross_entropy_with_logits加速计算
    #第一个参数是不包括softmax函数的前向传播结果，第二个是训练数据的正确答案
    #tf.argmax(y_,1)在每一行取最大下标
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=tf.argmax(y_,1))
    cross_entropy_mean = tf.reduce_mean(cross_entropy)

    #计算正则化损失，一般只计算在神经网络边权重上的正则化损失，而不使用偏置项
    regularizer = tf.contrib.layers.l2_regularizer(redularizetion_rate)
    regularization = regularizer(weights1) + regularizer(weights2)

    loss = cross_entropy_mean + regularization
    
    learning_rate = tf.train.exponential_decay(learning_rate_base, global_step,
                    mnist.train.num_examples, learning_rate_decay)
    
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)

    train_op = tf.group(train_step, variables_average_op)

    #计算预测答案和真实答案是否相等
    correct_prediction = tf.equal(tf.argmax(average_y,1), tf.argmax(y_,1))

    #tf.cast()将布尔型转化为实型
    accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
    #accuracy = tf.reduce_mean(correct_prediction)


    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)
        #准备验证数据
        validate_feed = {x: mnist.validation.images, y_: mnist.validation.labels}
        #准备测试数据
        test_feed = {x: mnist.test.images, y_:mnist.test.labels}

        for i in range(training_steps):
            if i % 1000 == 0:
                validate_accuracy = sess.run(accuracy, feed_dict=validate_feed)
                print("After %d training steps, validation accuracy using average model is %g" % 
                        (i,validate_accuracy))
                print(sess.run(learning_rate))
                # print(sess.run(y, feed_dict=validate_feed))
                # print(sess.run(y, feed_dict=validate_feed))
            
            xs, ys = mnist.train.next_batch(batch_size)
            sess.run(train_op, feed_dict={x: xs, y_:ys})
        
        test_accuracy = sess.run(accuracy, feed_dict=test_feed)
        print("After %d training steps, test accuracy using average model is %g" % 
                    (training_steps,test_accuracy))

        print(mnist.train.num_examples)

# def main(argv=None):
mnist = input_data.read_data_sets("/home/dz/tftest/tensorflow", one_hot=True)
train(mnist)
# if __name__ == 'main':
#   tf.app.run()
