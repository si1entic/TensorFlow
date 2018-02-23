# 屏蔽Warning:"Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX AVX2"
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
# 获取数据（如果存在就读取，不存在就下载完再读取）
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# 输入（张量，用占位符表示）
x = tf.placeholder("float", [None, 784])    # 任意数量的MNIST图像（None表示第一维任意长，第二维784即为28x28图片的像素点数）
y_ = tf.placeholder("float", [None, 10])    # 期望值（用10维向量来表示图片所表示数字0~9）

# 初始化Variable（可修改的张量，可用于计算，也可在计算中被修改）
W = tf.Variable(tf.zeros([784, 10]))    # 权重（用784维的图片向量乘它以得到一个10维的证据值向量）
b = tf.Variable(tf.zeros([10]))         # 偏置量（与10维的证据值向量相加）

# 计算模型结果
y = tf.nn.softmax(tf.matmul(x, W) + b)   # 输入矩阵x与权重矩阵W相乘，加上偏置矩阵b，然后求softmax

# 计算cost（这里用交叉熵来作为成本函数）
cross_entropy = -tf.reduce_sum(y_*tf.log(y))

# 使用梯度下降法（步长0.01），来使cost最小
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

# 在Session里启动模型，并初始化变量
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

# 进行随机梯度下降训练，这个过程中输入数字的图片和期望，权重和偏置量会朝着使cost最小的方向改变
for i in range(1000):    # 训练1000次
    batch_xs, batch_ys = mnist.train.next_batch(100)             # 随机获取100张MNIST图片的image和label
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})  # 用batch_xs替换x，batch_ys替换y_来执行train_step

# 计算训练精度
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))      # 比对预测标签和实际标签是否匹配
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))     # 将一组布尔值转化为浮点数
print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))  # 输出学习所得模型在测试集上的正确率
