import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# 读入数据集
mnist = input_data.read_data_sets('MNIST_data/', one_hot=True)

x_ = tf.placeholder(tf.float32, [None, 784])
y_ = tf.placeholder(tf.float32, [None, 10])

# 将图片从784维向量还原为28x28的矩阵图片
x_image = tf.reshape(x_, [-1, 28, 28, 1])


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape)
    return tf.Variable(initial)


def conv2d(x, w):
    return tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


# 第一层卷积
W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

# 第二层卷积
W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

# 第一层全连接，输出为1024维的向量
W_fc1 = weight_variable([7*7*64, 1024])
b_fc1 = bias_variable([1024])
h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
# 使用dropout，keep_prob是一个占位符，训练时为0.5，测试时为1
keep_drop = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_drop)

# 第二层全连接，把1024维的向量转换为10维，对应10个类别
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])
y = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

# 定义交叉熵损失
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
# 定义训练过程train_step
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

# 定义准确率
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

# 创建Session和变量初始化
sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())

# 训练20000步，每步50张图片
for i in range(20000):
    batch = mnist.train.next_batch(50)
    # 每1000步打印一次在验证集上的准确率
    if i % 1000 == 0:
        train_accuracy = accuracy.eval(
            feed_dict={x_: mnist.validation.images, y_: mnist.validation.labels, keep_drop: 1.0})
        print('step %d, training accuracy %g' % (i, train_accuracy))
    train_step.run(feed_dict={x_: batch[0], y_: batch[1], keep_drop: 0.5})

# 训练结束，打印模型在测试集上的准确率
print('test accuracy %g' % accuracy.eval(
    feed_dict={x_: mnist.test.images, y_: mnist.test.labels, keep_drop: 1.0}))
