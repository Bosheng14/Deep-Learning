import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# 读入数据集
mnist = input_data.read_data_sets('MNIST_data/', one_hot=True)

# 定义输入x，输出y，模型参数W，模型参数b的占位符，None表示任意多个
x = tf.placeholder(tf.float32, [None, 784])
y_ = tf.placeholder(tf.float32, [None, 10])
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

# 计算SoftMax模型输出
y = tf.nn.softmax(tf.matmul(x, W) + b)

# 定义交叉熵损失
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y)))
# 定义训练过程train_step
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

# 创建Session和变量初始化
sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

# 训练1000步，每步100张图片
for _ in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

# 定义准确率
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# 训练结束，打印模型在测试集上的准确率
print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
