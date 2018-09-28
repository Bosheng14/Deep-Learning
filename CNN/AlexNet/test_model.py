import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
from model import AlexNet
from caffe_classes import class_names


class AlexNetTest(object):

    def __init__(self):
        self.PRE_TRAINED_WEIGHTS = 'bvlc_alexnet.npy'
        self.NUM_CLASSES = 1000

    def test(self):
        # ImageNet数据集BGR均值
        image_mean = np.array([104., 117., 124.], dtype=np.float32)

        image_dir = os.path.join(os.getcwd(), 'images')
        # 获取images文件夹下所有图片的列表
        img_files = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith('.jpeg')]

        images = []
        skip_layer = []
        for f in img_files:
            images.append(cv2.imread(f))

        # 显示图片
        fig = plt.figure(figsize=(15, 6))
        for i, img in enumerate(images):
            fig.add_subplot(1, 3, i+1)
            plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            plt.axis('off')

        # 创建AlexNet模型
        x = tf.placeholder(tf.float32, [1, 227, 227, 3])
        keep_prob = tf.placeholder(tf.float32)
        model = AlexNet(x, keep_prob, self.NUM_CLASSES, skip_layer, weights_path=self.PRE_TRAINED_WEIGHTS)
        score = model.fc8
        softmax = tf.nn.softmax(score)

        # 创建TensorFlow Session会话
        with tf.Session() as sess:

            sess.run(tf.global_variables_initializer())

            model.load_initial_weights(sess)

            fig2 = plt.figure(figsize=(15, 6))
            for i, image in enumerate(images):
                # Convert image to float32 and resize to (227x227)
                img = cv2.resize(image.astype(np.float32), (227, 227))
                # 减去均值
                img -= image_mean
                # reshape为模型指定的输入形式
                img = img.reshape((1, 227, 227, 3))
                # 计算分类的概率
                predict_prob = sess.run(softmax, feed_dict={x: img, keep_prob: 1.0})
                # 根据最大概率值获取类目的名称
                class_name = class_names[np.argmax(predict_prob)]

                fig2.add_subplot(1, 3, i+1)
                plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                plt.title('Class: ' + class_name + ', probability: %.4f' % predict_prob[0, np.argmax(predict_prob)])
                plt.axis('off')
        plt.show()


def main():
    if not os.path.exists(os.path.join(os.getcwd(), 'images')):
        print(' [***] Loading test images Error!!')
    else:
        alex_test = AlexNetTest()
        alex_test.test()


if __name__ == '__main__':
    main()


