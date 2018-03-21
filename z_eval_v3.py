import platform
import tensorflow as tf
import os
import datasets_reader as dr
import matplotlib.pyplot as plt
from multiprocessing import cpu_count
import alex2_inference as net_inference
import time
from datetime import datetime
import math
import get_input
import numpy as np
import sys

BATCH_SIZE = 128
LEARNING_RATE_BASE = 0.000001
LEARNING_RATE_DECAY = 0.99
REGULARIZATION_RATE = 0.0001
TRAINING_STEPS = 140000
MOVING_AVERAGE_DECAY = 0.99
DATA_FRAMES_PATH = []
IN_NET_SIZE = [224, 224]

if platform.system() == 'Windows':
    N_GPU = [0]
    DATA_PATH = r'D:\Desktop\DEMO_FILE\cifar-10-batches-py'
    MODEL_SAVE_PATH = r'D:\Desktop\DEMO_FILE\cifar10_model'
    MODEL_NAME = "model.ckpt"
elif platform.system() == 'Linux':
    N_GPU = [0, 1, 2, 3]
    DATA_PATH = '/home/zbh/Desktop/alpha_2_zbh/cifar-10-batches-py'
    MODEL_SAVE_PATH = '/home/zbh/Desktop/alpha_1_zbh/cifar10_model'
    MODEL_NAME = "model.ckpt"
    cifar_tutorial = '/home/zbh/Desktop/alpha_2_zbh/tensorflow_org/models/tutorials/image/cifar10'
    cifar10_data_path = '/home/zbh/Desktop/datasets/cifar-10/cifar-10-binary/cifar-10-batches-bin'
    sys.path.append(cifar_tutorial)
    import cifar10_input
    import cifar10
else:
    exit('ZBH: unknown platform system %s.' % platform.system())


def evaluate():

    with tf.Graph().as_default() as g:
        cifar10.maybe_download_and_extract()
        eval_data = 'test'
        images, labels = cifar10.inputs(eval_data=eval_data)

        logits = cifar10.inference(images)

        # with tf.variable_scope(tf.get_variable_scope()):
        #     label_batch_ = net_inference.inference(image_batch, 3, len(class_list), None, False, net_stream='spatial')
        # variable_averages = tf.train.ExponentialMovingAverage(net_train.MOVING_AVERAGE_DECAY)
        # variables_to_restore = variable_averages.variables_to_restore()
        # saver = tf.train.Saver(variables_to_restore)
        saver = tf.train.Saver()
        os.environ["CUDA_VISIBLE_DEVICES"] = '3'
        gpu_options = tf.GPUOptions(allow_growth=True)
        config = tf.ConfigProto(allow_soft_placement=True, gpu_options=gpu_options)
        while True:
            with tf.Session(config=config) as sess:
                coord = tf.train.Coordinator()
                threads = tf.train.start_queue_runners(sess=sess, coord=coord)


                # correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
                # accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

                ckpt = tf.train.get_checkpoint_state(MODEL_SAVE_PATH)
                if ckpt and ckpt.model_checkpoint_path:
                    saver.restore(sess, ckpt.model_checkpoint_path)
                    global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
                    whole = []
                    label_all_ground = []
                    label_all_eval = []
                    batch_go_on = True
                    while batch_go_on is True:
                        aa, bb = sess.run([tf.argmax(logits, 1), labels])
                        # temp = list(zip(aa, bb, cc))
                        # whole = whole + temp
                        #
                        # label_all_ground = label_all_ground + list(ee)
                        # label_all_eval = label_all_eval + list(dd)
                        #
                        # if whole[0] in whole[-net_train.BATCH_SIZE:] and len(whole) > net_train.BATCH_SIZE:
                        #     whole_end = whole.index(whole[0], -net_train.BATCH_SIZE)
                        #     whole = whole[:whole_end]
                        #     label_all_ground = label_all_ground[:whole_end]
                        #     label_all_eval = label_all_eval[:whole_end]
                        #     batch_go_on = False
                        print('*****************')
                        print(aa)
                        print(bb)
                    print("After %s training step(s), validation accuracy = 100" % (global_step))
                else:
                    print('No checkpoint file found')
                    return

                coord.request_stop()
                coord.join(threads)

                time.sleep(10)


def main():
    evaluate()


if __name__ == '__main__':
    main()
