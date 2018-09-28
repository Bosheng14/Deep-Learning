import os
import argparse
import numpy as np
import tensorflow as tf
from datetime import datetime
from model import AlexNet
from data_generator import ImageDataGenerator


def parse_args():
    """
    解析命令行参数
    """
    parser = argparse.ArgumentParser(description='Fine-Tuning Arguments')
    parser.add_argument('--train_files', type=str, default='./train_set/train.txt', help='Txt files for train')
    parser.add_argument('--val_files', type=str, default='./validation_set/val.txt', help='Txt files for validation')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='initial learning rate')
    parser.add_argument('--num_epochs', type=int, default=1000, help='training epoch number')
    parser.add_argument('batch_size', type=int, default=128, help='training batch size')
    parser.add_argument('keep_pro', type=float, default=0.5, help='keep probability for dropout')
    parser.add_argument('num_classes', type=int, default=2, help='classification number')
    parser.add_argument('train_layers', type=list, default=['fc7', 'fc8'], help='layers for fine-tuning')
    parser.add_argument('display_step', type=int, default=10, help='steps for display')
    parser.add_argument('summary_writer_path', type=str, default='./train_log/', help='path for summary files')
    parser.add_argument('checkpoints_path', type=str, default='./checkpoints/', help='path for checkpoints files')
    args = parser.parse_args()

    return args


class AlexNetFineTune(object):

    def __init__(self, train_files, val_files, learning_rate, num_epochs,
                 batch_size, keep_pro, num_classes, train_layers, display_step,
                 summary_writer_path, checkpoints_path):
        self.PRE_TRAINED_WEIGHTS = 'bvlc_alexnet.npy'
        self.LEARNING_RATE = learning_rate
        self.TRAIN_FILES = train_files
        self.VAL_FILES = val_files
        self.NUM_EPOCHS = num_epochs
        self.BATCH_SIZE = batch_size
        self.KEEP_PRO = keep_pro
        self.NUM_CLASSES = num_classes
        self.TRAIN_LAYERS = train_layers
        self.DISPLAY_STEP = display_step
        self.SUMMARY_WRITER_PATH = summary_writer_path
        self.CHECKPOINTS_PATH = checkpoints_path

    def fine_tuning(self):

        x = tf.placeholder(tf.float32, [self.BATCH_SIZE, 227, 227, 3])
        y = tf.placeholder(tf.float32, [None, self.NUM_CLASSES])
        keep_prob = tf.placeholder(tf.float32)

        model = AlexNet(x, keep_prob, self.NUM_CLASSES, self.TRAIN_LAYERS)

        score = model.fc8

        var_list = [v for v in tf.trainable_variables() if v.name.split('/')[0] in self.TRAIN_LAYERS]

        # Op for calculate the loss
        with tf.name_scope('cross_ent'):
            loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=score, labels=y))

        # Op for train
        with tf.name_scope('train'):
            # 获取训练变量的梯度
            gradients = tf.gradients(loss, var_list)
            gradients = list(zip(gradients, var_list))

            optimizer = tf.train.GradientDescentOptimizer(self.LEARNING_RATE)
            train_op = optimizer.apply_gradients(grads_and_vars=gradients)

        # 添加gradients、var、loss的summary
        for gradient, var in gradients:
            tf.summary.histogram(var.name + '/gradient', gradient)
        for var in var_list:
            tf.summary.histogram(var.name, var)
        tf.summary.scalar('cross_entropy', loss)

        # Evaluation Op: accuracy of the model
        with tf.name_scope('accuracy'):
            correct_pre = tf.equal(tf.argmax(score, 1), tf.argmax(y, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_pre, tf.float32))

        # 添加accuracy的summary
        tf.summary.scalar('accuracy', accuracy)

        # Merge all summaries together and initialize the FileWriter
        merged_summary = tf.summary.merge_all()
        writer = tf.summary.FileWriter(self.SUMMARY_WRITER_PATH)

        # 创建tf.train.Saver()用于存储checkpoints文件
        saver = tf.train.Saver()
        train_generator = ImageDataGenerator(self.TRAIN_FILES, horizontal_flip=True, shuffle=True)
        val_generator = ImageDataGenerator(self.VAL_FILES, shuffle=False)

        # 计算每个epoch所需的step
        train_batches_per_epoch = np.floor(train_generator.data_size / self.BATCH_SIZE).astype(np.int16)
        val_batches_per_epoch = np.floor(val_generator.data_size / self.BATCH_SIZE).astype(np.int16)

        # 创建tf.Session()会话
        with tf.Session() as sess:

            sess.run(tf.global_variables_initializer())

            writer.add_graph(sess.graph)

            model.load_initial_weights(sess)

            print('{} Start training...'.format(datetime.now()))
            print('{} Open TensorBoard at --train_log {}'.format(datetime.now(), self.SUMMARY_WRITER_PATH))

            for epoch in range(self.NUM_EPOCHS):
                print('{} Epoch number: {}'.format(datetime.now(), epoch+1))
                step = 1
                while step < train_batches_per_epoch:
                    batch_xs, batch_ys = train_generator.next_batch(self.BATCH_SIZE)
                    sess.run(train_op, feed_dict={x: batch_xs, y: batch_ys, keep_prob: self.KEEP_PRO})

                    if step % self.DISPLAY_STEP == 0:
                        s = sess.run(merged_summary, feed_dict={x: batch_xs, y: batch_ys, keep_prob: 1.})
                        writer.add_summary(s, epoch * train_batches_per_epoch + step)
                    step += 1

                print('{} Start validation...'.format(datetime.now()))
                test_acc = 0
                test_count = 0
                for _ in range(val_batches_per_epoch):
                    batch_ts, batch_ty = val_generator.next_batch(self.BATCH_SIZE)
                    acc = sess.run(accuracy, feed_dict={x: batch_ts, y: batch_ty, keep_prob: 1.})
                    test_acc += acc
                    test_count += 1
                test_acc /= test_count
                print('Validation Accuracy = {:.4f}'.format(datetime.now(), test_acc))

                val_generator.reset_pointer()
                train_generator.reset_pointer()

                # 保存checkpoints文件
                print('{} Saving checkpoints of model...'.format(datetime.now()))
                if epoch % 50 == 0:
                    checkpoints_name = os.path.join(self.CHECKPOINTS_PATH, 'model_epoch'+str(epoch)+'.ckpt')
                    save_path = saver.save(sess, checkpoints_name)
                    print('{} Model checkpoints saved at {}'.format(datetime.now(), checkpoints_name))


def main():
    args = parse_args()

    if not os.path.exists(args.checkpoints_path):
        os.mkdir(args.checkpoints_path)

    if not os.path.exists(args.summary_writer_path):
        os.mkdir(args.summary_writer_path)

    if not (os.path.exists(args.train_files) and os.path.exists(args.val_files)):
        print(' [***] There should be Txt files for train and validation.')

    else:
        alex_fine_tuning = AlexNetFineTune(args)
        alex_fine_tuning.fine_tuning()


if __name__ == '__main__':
    tf.app.run()
