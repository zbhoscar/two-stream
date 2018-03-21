# -*- coding: UTF-8 -*- #
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


def main():
    with tf.Graph().as_default(), tf.device('/cpu:0'):
        # _, train = get_input.get_cifar10_input(DATA_PATH)

        images_train, labels_train = cifar10_input.distorted_inputs(data_dir=cifar10_data_path, batch_size=BATCH_SIZE)
        # image_test, labels_test = cifar10_input.distorted_inputs(eval_data=True, data_dir=cifar10_data_path,
        #                                                          batch_size=BATCH_SIZE)

        # image_holder = tf.placeholder(tf.float32, [BATCH_SIZE, 24, 24, 3])
        # label_holder = tf.placeholder(tf.int32, [BATCH_SIZE])
        # resized_x = tf.image.resize_images(x, IN_NET_SIZE, method=0)
        # regu = tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)
        regu = None
        global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False)
        # learning_rate = tf.train.exponential_decay(LEARNING_RATE_BASE, global_step,
        #                                            TRAINING_STEPS / math.log(0.005, LEARNING_RATE_DECAY),
        #                                            LEARNING_RATE_DECAY)
        num_batcher_per_epoch = 50000 / BATCH_SIZE
        decay_step = int(num_batcher_per_epoch * 350.0)
        learning_rate = tf.train.exponential_decay(0.01, global_step, decay_step, 0.1)
        opt = tf.train.GradientDescentOptimizer(learning_rate)

        cur_loss = []
        tower_grads = []
        reuse_variables = False
        for i in N_GPU:
            with tf.device('/gpu:%d' % i):
                with tf.name_scope('GPU_%d' % i) as scope:
                    loss = tower_loss(scope,images_train, labels_train, reuse_variables)
                    tf.get_variable_scope().reuse_variables()

                    # cur_loss = get_loss(images_train, labels_train, regu, 10, scope, reuse_variables)
                    # reuse_variables = True
                    # # tf.get_variable_scope().reuse_variables()


                    grads = opt.compute_gradients(loss)
                    tower_grads.append(grads)

        grads = average_gradients(tower_grads)

        for grad, var in grads:
            if grad is not None:
                tf.summary.histogram('gradients_on_average/%s' % var.op.name, grad)

        apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

        for var in tf.trainable_variables():
            tf.summary.histogram(var.op.name, var)

        # variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
        # variables_to_average = (tf.trainable_variables() + tf.moving_average_variables())
        # variables_averages_op = variable_averages.apply(variables_to_average)

        # train_op = tf.group(apply_gradient_op, variables_averages_op)

        saver = tf.train.Saver(tf.all_variables())
        summary_op = tf.summary.merge_all()

        # top_k_op = tf.nn.in_top_k()

        init = (tf.global_variables_initializer(), tf.local_variables_initializer())
        gpu_options = tf.GPUOptions(allow_growth=True)
        config = tf.ConfigProto(allow_soft_placement=True, gpu_options=gpu_options)
        with tf.Session(config=config) as sess:

            sess.run(init)

            # if os.path.exists(MODEL_SAVE_PATH):
            #     ckpt = tf.train.get_checkpoint_state(MODEL_SAVE_PATH)  # net_train.MODEL_SAVE_PATH)
            #     # num_str = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
            #     saver_goon = tf.train.Saver()
            #     saver_goon.restore(sess, ckpt.model_checkpoint_path)


            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)
            # tf.train.start_queue_runners()

            summary_writer = tf.summary.FileWriter(MODEL_SAVE_PATH, sess.graph)

            for step in range(TRAINING_STEPS):
                # sta = (step * BATCH_SIZE) % 50000
                # end = (step * BATCH_SIZE + BATCH_SIZE) % 50000
                # if sta < end:
                #     xs = train[b'data'][sta:end]
                #     ys = train[b'labels'][sta:end]
                # else:
                #     xs = np.row_stack((train[b'data'][sta:], train[b'data'][:end]))
                #     ys = train[b'labels'][sta:] + train[b'labels'][:end]

                # ys = np.array(ys)
                # print("*****************************")
                # print(type(ys))
                # print("*****************************")
                start_time = time.time()
                # image_batch, label_batch = sess.run([images_train, labels_train])
                # print(image_batch, image_batch.dtype, image_batch.shape)
                # print(label_batch, label_batch.dtype, label_batch.shape)

                _, loss_value = sess.run([apply_gradient_op, loss])#, train_op, cur_loss
                                         #feed_dict={image_holder: image_batch, label_holder: label_batch})
                duration = time.time() - start_time
                if step != 0 and step % 10 == 0:
                    num_examples_per_step = BATCH_SIZE * len(N_GPU)
                    examples_per_sec = num_examples_per_step / duration
                    sec_per_batch = duration / len(N_GPU)
                    format_str = '%s: step %d, loss = %.2f (%.1f examples/sec; %.3f sec/batch)'
                    print(format_str % (datetime.now(), step, loss_value, examples_per_sec, sec_per_batch))
                    summary = sess.run(summary_op)
                    summary_writer.add_summary(summary, step)
                if step % 1000 == 0 or (step + 1) == TRAINING_STEPS:
                    checkpoint_path = os.path.join(MODEL_SAVE_PATH, MODEL_NAME)
                    saver.save(sess, checkpoint_path, global_step=step)

                # a,c = sess.run([image_batch, label_batch])
                # #print(a,a.shape)
                # print(c,c.shape,c.dtype)

            coord.request_stop()
            coord.join(threads)


def tower_loss(scope, images, labels, reuse_variables):
    logits = cifar10.inference(images)
    _ = cifar10.loss(logits, labels)
    losses = tf.get_collection('losses', scope)
    total_loss = tf.add_n(losses, name='total_loss')
    return total_loss


def get_loss(x, y_, regu, class_num, scope, reuse_variables=None):
    # y_ = tf.cast(y_, tf.int64)
    # (input_batch, input_channel_num, regu, net_stream='spatial')
    with tf.variable_scope(tf.get_variable_scope(), reuse=reuse_variables):
        y = net_inference.inference(x, 3, class_num, regu, True, 'spatial')
    cross_entropy = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=y_))
    # regularization_loss = tf.add_n(tf.get_collection('spatial_loss', scope))
    loss = cross_entropy  # + regularization_loss
    return loss


def average_gradients(tower_grads):
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        grads = []
        for g, _ in grad_and_vars:
            expanded_g = tf.expand_dims(g, 0)
            grads.append(expanded_g)

        grad = tf.concat(axis=0, values=grads)
        grad = tf.reduce_mean(grad, 0)
        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)

    return average_grads


if __name__ == '__main__':
    main()
