# /*
#  * @Author: thunder_dz
#  * @Date: 2019-03-23 17:07:00
#  * @Last Modified by:   thunder_dz
#  * @Last Modified time: 2019-03-23 17:07:00
#  */


#书P109页
import tensorflow as tf
labels = [[0,0,1],[0,1,0]]
labels2 = [[0.4,0.1,0.5],[0.3,0.6,0.1]]
labels3 = [2,1]
logits = [[2,  0.5,6],
          [0.1,0,  3]]

logits_scaled = tf.nn.softmax(logits)
logits_scaled2 = tf.nn.softmax(logits_scaled)


#求softmax和交叉熵合在一个地方，对one-hot类型进行使用
result1 = tf.nn.softmax_cross_entropy_with_logits(labels=labels,logits=logits)
loss = tf.reduce_mean(result1)
result2 = tf.nn.softmax_cross_entropy_with_logits(labels=labels,logits=logits_scaled)

#矩阵按行求和
result3 = -tf.reduce_sum(labels*tf.log(logits_scaled),1)

result4 = tf.nn.softmax_cross_entropy_with_logits(labels=labels2,logits=logits)

#这个是对非one-hot类型进行使用
result5 = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels3,logits=logits)

with tf.Session() as sess:
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    print ("scaled = ",sess.run(logits_scaled))
    print ("scaled2 = ",sess.run(logits_scaled2))
    print ("result1 = ",sess.run(result1))
    print ("loss = ",sess.run(loss))
    print ("result2 = ",sess.run(result2))
    print ("result3 = ",sess.run(result3))   #验证result1的正确性
    #比较result1和result4发现，对于正确分类的交叉熵和错误分类的交叉熵
    #二者的结果没有标准one-hot那么明显
    print ("result4 = ",sess.run(result4))
    #对非one-hot的标签进行交叉熵计算
    print("result5 = ",sess.run(result5))