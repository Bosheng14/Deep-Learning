import numpy as np
import cv2


class ImageDataGenerator:

    def __init__(self, class_list, horizontal_flip=False, shuffle=False,
                 mean=np.array([104., 117., 124.]), scale_size=(227, 227), nb_classes=2):

        self.horizontal_flip = horizontal_flip
        self.n_classes = nb_classes
        self.shuffle = shuffle
        self.mean = mean
        self.scale_size = scale_size
        self.pointer = 0

        self.read_class_list(class_list)

        if self.shuffle:
            self.shuffle_data()

    def read_class_list(self, class_list):
        """
        读取图像文件列表，获取图像及其对应标签，图像文件列表必须提前制定
        """
        with open(class_list) as f:
            lines = f.readlines()
            self.images = []
            self.labels = []
            for l in lines:
                items = l.split()
                self.images.append(items[0])
                self.labels.append(int(items[1]))

            self.data_size = len(self.labels)

    def shuffle_data(self):
        """
        随机shuffle图像及其标签
        """
        images = self.images.copy()
        labels = self.labels.copy()
        self.images = []
        self.labels = []

        # 生成随机序列并根据序列进行data shuffle
        idx = np.random.permutation(len(labels))
        for i in idx:
            self.images.append(images[i])
            self.labels.append(labels[i])

    def reset_pointer(self):
        """
        重置列表指示位置
        """
        self.pointer = 0

        if self.shuffle:
            self.shuffle_data()

    def next_batch(self, batch_size):
        """
        获取下一个batch_size并将图像标签转换为ont-hot编码
        """
        paths = self.images[self.pointer:self.pointer + batch_size]
        labels = self.labels[self.pointer:self.pointer + batch_size]

        self.pointer += batch_size

        # 读取图像
        images = np.ndarray([batch_size, self.scale_size[0], self.scale_size[1], 3])
        for i in range(len(paths)):
            img = cv2.imread(paths[i])

            # 随机水平翻转图像
            if self.horizontal_flip and np.random.random() < 0.5:
                # cv2.flip(img, n)函数对原始图像进行翻转操作，0垂直翻转，1水平翻转，-1水平垂直翻转
                img = cv2.flip(img, 1)

            # 缩放图像
            img = cv2.resize(img, (self.scale_size[0], self.scale_size[1]))
            img = img.astype(np.float32)

            img -= self.mean

            images[i] = img

        # 将图像标签转换为one-hot编码
        one_hot_labels = np.zeros((batch_size, self.n_classes))
        for i in range(len(labels)):
            one_hot_labels[i][labels[i]] = 1

        return images, one_hot_labels
