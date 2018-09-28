import tensorflow as tf
import numpy as np


def max_pooling(x, kernel_h, kernel_w, stride_h, stride_w, name, padding='SAME'):
    return tf.nn.max_pool(x, ksize=[1, kernel_h, kernel_w, 1],
                          strides=[1, stride_h, stride_w, 1], padding=padding, name=name)


def lrn(x, depth_radius, alpha, beta, name, bias=1.0):
    return tf.nn.local_response_normalization(x, depth_radius=depth_radius,
                                              alpha=alpha, beta=beta, bias=bias, name=name)


def dropout(x, keep_prob):
    return tf.nn.dropout(x, keep_prob)


def fc_layer(x, in_division, out_division, name, re_flag=True):
    with tf.variable_scope(name) as scope:
        # tf.Variable(name=None, initial_value, validate_shape=True, trainable=True, collections=None)
        # tf.get_variable(name, shape=None, initializer=None, dtype=tf.float32, trainable=True, collections=None)
        # 二者区别：
        #   相同点：都可以用来创建或者获取一个变量
        #   不同点：二者在创建变量时功能基本上是等价的。最大的区别在于tf.Variable的变量名是一个可选项，可通过name参数给出
        #          而tf.get_variable必须指定变量名。
        # TensorFlow中获取变量的机制主要是通过tf.get_variable和tf.variable_scope实现的，当reuse=False或者None(默认值)时，
        # 同一个tf.variable_scope下的变量名不能相同；当reuse=True时，tf.variable_scope只能获取创建过的变量。
        w = tf.get_variable('weights', shape=[in_division, out_division], trainable=True)
        b = tf.get_variable('biases', shape=[out_division], trainable=True)

        fc_out = tf.nn.xw_plus_b(x, w, b, name=scope.name)

        if re_flag:
            return tf.nn.relu(fc_out)
        else:
            return fc_out


def conv_layer(x, kernel_h, kernel_w, num_features, stride_h, stride_w, name, padding='SAME', groups=1):

    """ 分组的卷积层，当groups=2时即可实现AlexNet中的分为上下两部分的结构 """

    input_channels = int(x.get_shape()[-1])

    convolve = lambda i, k: tf.nn.conv2d(i, k, strides=[1, stride_h, stride_w, 1], padding=padding)

    with tf.variable_scope(name) as scope:
        w = tf.get_variable('weights', shape=[kernel_h, kernel_w, input_channels/groups, num_features])
        b = tf.get_variable('biases', shape=[num_features])

    if groups == 1:
        conv = convolve(x, w)
    else:
        # 对输入(x)、权重(W)进行分组并分别卷积
        x_groups = tf.split(axis=3, num_or_size_splits=groups, value=x)
        w_groups = tf.split(axis=3, num_or_size_splits=groups, value=w)
        output_groups = [convolve(i, k) for i, k in zip(x_groups, w_groups)]

        # 合并分组卷积的结果
        conv = tf.concat(axis=3, values=output_groups)

    # tf.shape(a)和a.get_shape()比较：
    #       相同点：都能得到tensor a 的尺寸
    #       不同点：tf.shape()中a的数据类型可以是tensor,list,array。而a.get_shape()中a的数据类型只能是tensor，且返回的是一个元组
    conv_out = tf.reshape(tf.nn.bias_add(conv, b), tf.shape(conv))

    return tf.nn.relu(conv_out, name=scope.name)


class AlexNet(object):

    def __init__(self, x, keep_prob, num_classes, skip_layer, weights_path='DEFAULT'):
        self.X = x
        self.NUM_CLASSES = num_classes
        self.KEEP_PROB = keep_prob
        self.SKIP_LAYER = skip_layer

        if weights_path == 'DEFAULT':
            self.WEIGHTS_PATH = 'bvlc_alexnet.npy'
        else:
            self.WEIGHTS_PATH = weights_path

        self.inference()

    def inference(self):

        """构建AlexNet网络结构，包含5个卷积层，3个全连接层"""

        # 1st Layer: Conv (w ReLu) -> Lrn -> Pool
        conv1 = conv_layer(self.X, 11, 11, 96, 4, 4, padding='VALID', name='conv1')
        pool1 = max_pooling(conv1, 3, 3, 2, 2, padding='VALID', name='pool1')
        norm1 = lrn(pool1, 2, 1e-5, 0.75, name='norm1')

        # 2nd Layer: Conv (w ReLu) -> Lrn -> Pool with groups=2
        conv2 = conv_layer(norm1, 5, 5, 256, 1, 1, groups=2, name='conv2')
        pool2 = max_pooling(conv2, 3, 3, 2, 2, padding='VALID', name='pool2')
        norm2 = lrn(pool2, 2, 1e-5, 0.75, name='norm2')

        # 3rd Layer: Conv (w ReLu)
        conv3 = conv_layer(norm2, 3, 3, 384, 1, 1, name='conv3')

        # 4th Layer: Conv (w ReLu) with groups=2
        conv4 = conv_layer(conv3, 3, 3, 384,  1, 1, groups=2, name='conv4')

        # 5th Layer: Conv (w ReLu) -> Pool with groups=2
        conv5 = conv_layer(conv4, 3, 3, 256, 1, 1, groups=2, name='conv5')
        pool5 = max_pooling(conv5, 3, 3, 2, 2, padding='VALID', name='pool5')

        # 6th Layer: Flatten -> FC (w ReLu) -> Dropout
        flattened = tf.reshape(pool5, shape=[-1, 6*6*256])
        fc6 = fc_layer(flattened, 6*6*256, 4096, name='fc6')
        dropout6 = dropout(fc6, self.KEEP_PROB)

        # 7th Layer: FC (w ReLu) -> Dropout
        fc7 = fc_layer(dropout6, 4096, 4096, name='fc7')
        dropout7 = dropout(fc7, self.KEEP_PROB)

        # 8th Layer: FC and return unscaled activations
        self.fc8 = fc_layer(dropout7, 4096, self.NUM_CLASSES, re_flag=False, name='fc8')

    def load_initial_weights(self, session):
        """ 加载预先训练好的权重参数 """

        # Load the weights into memory，训练好的参数是以字典列表的形式存储在bvlc_alexnet.npy文件中
        weights_dict = np.load(self.WEIGHTS_PATH, encoding='bytes').item()

        for op_name in weights_dict:
            if op_name not in self.SKIP_LAYER:
                with tf.variable_scope(op_name, reuse=True):
                    for data in weights_dict[op_name]:
                        if len(data.shape) == 1:
                            var = tf.get_variable('biases', trainable=False)
                            session.run(var.assign(data))
                        else:
                            var = tf.get_variable('weights', trainable=False)
                            session.run(var.assign(data))
