# -*- coding: UTF-8 -*- #
import platform
import tensorflow as tf
import os
import z_input
import time
from datetime import datetime
import z_inference
import sys

# import datasets_reader as dr
# from multiprocessing import cpu_count

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_integer('batch_size', 128, None)
tf.app.flags.DEFINE_boolean('bool_eval', False, None)
tf.app.flags.DEFINE_boolean('demo', False, None)
tf.app.flags.DEFINE_string('key', 'UCF101', None)
tf.app.flags.DEFINE_string('model_name', 'model.ckpt', None)
tf.app.flags.DEFINE_integer('of_stack_num', 5, None)
# ### EMPTY VARS, CHANGED IN INPUTTING ###
tf.app.flags.DEFINE_integer('num_class', 0, None)
tf.app.flags.DEFINE_string('model_save_path', '', None)
# ### DEPENDS ON SYSTEM ###
if platform.system() == 'Windows':
    N_GPU = [0]
    tf.app.flags.DEFINE_string('root_path', r'D:\Desktop\DEMO_FILE\UCF101pic_256',
                               None)  # where to find pre-processed video
    tf.app.flags.DEFINE_string('cousin_path', r'..\..\Deep_Learning_with_TensorFlow\tf_models', None)
elif platform.system() == 'Linux':
    N_GPU = [0, 1, 2, 3]
    tf.app.flags.DEFINE_string('root_path', '/home/zbh/Desktop/alpha_2_zbh/UCF101pic_256', None)
    tf.app.flags.DEFINE_string('cousin_path', '../../alpha_1_zbh/tf_models', None)
else:
    exit('ZBH: unknown platform system %s.' % platform.system())

LEARNING_RATE_BASE = 0.000001
LEARNING_RATE_DECAY = 0.99
REGULARIZATION_RATE = 0.0001
TRAINING_STEPS = 1000000
MOVING_AVERAGE_DECAY = 0.99


# cifar_tutorial = '/home/zbh/Desktop/alpha_2_zbh/tensorflow_org/models/tutorials/image/cifar10'
# cifar10_data_path = '/home/zbh/Desktop/datasets/cifar-10/cifar-10-binary/cifar-10-batches-bin'
# sys.path.append(cifar_tutorial)
# import cifar10_input
# import cifar10


def main():
    with tf.Graph().as_default(), tf.device('/cpu:0'):

        # flags = FLAGS
        # image_batch, label_batch = cifar10_input.distorted_inputs(data_dir=cifar10_data_path,
        #                                                           batch_size=flags.batch_size)
        # flags.model_save_path = '/home/zbh/Desktop/alpha_1_zbh/cifar10_model'
        # flags.num_class = 10
        image_batch, of_stack_batch, label_batch, rootb, nameb, indxb, flags = \
            z_input.get_2stream_input(FLAGS, of_stack_num=FLAGS.of_stack_num)
        # SET LEARNING STRATEGY
        global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False)
        # regu = None  # tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)
        learning_rate = tf.train.exponential_decay(0.001, global_step, 80000, 0.2, staircase=True)
        opt = tf.train.GradientDescentOptimizer(learning_rate)

        # MULTI_GPU to calculate loss and train_op
        loss = []
        tower_grads = []
        for i in N_GPU:
            with tf.device('/gpu:%d' % i):
                with tf.name_scope('GPU_%d' % i) as scope:
                    loss = tower_loss(scope, image_batch, of_stack_batch, label_batch, flags)
                    tf.get_variable_scope().reuse_variables()
                    grads = opt.compute_gradients(loss)
                    tower_grads.append(grads)
        grads = average_gradients(tower_grads)
        train_op = opt.apply_gradients(grads, global_step=global_step)

        # Persistence training
        for grad, var in grads:
            if grad is not None:
                tf.summary.histogram('gradients_on_average/%s' % var.op.name, grad)
        for var in tf.trainable_variables():
            tf.summary.histogram(var.op.name, var)
        saver = tf.train.Saver(tf.all_variables())
        summary_op = tf.summary.merge_all()

        # Begin training
        init = (tf.global_variables_initializer(), tf.local_variables_initializer())
        gpu_options = tf.GPUOptions(allow_growth=True)
        config = tf.ConfigProto(allow_soft_placement=True, gpu_options=gpu_options)
        with tf.Session(config=config) as sess:

            sess.run(init)
            #     if os.path.exists(MODEL_SAVE_PATH):
            #         ckpt = tf.train.get_checkpoint_state(MODEL_SAVE_PATH)#net_train.MODEL_SAVE_PATH)
            #         # num_str = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
            #         saver_goon = tf.train.Saver()
            #         saver_goon.restore(sess, ckpt.model_checkpoint_path)
            #     else:
            #         sess.run(init)

            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)

            # #### INPUT CHECK ####
            # import matplotlib.pyplot as plt
            # for i in range(3):
            #     a, b, c, d, e, f = sess.run([image_batch, of_stack_batch, label_batch, rootb, nameb, indxb, ])
            #     print(a, a.shape, c.dtype)
            #     print(b, b.shape, b.dtype)
            #     print(c, c.shape, c.dtype)
            #     idd = 0
            #     print(c[idd], d[idd], e[idd], f[idd])
            #     plt.imshow(a[idd, :, :, :])
            #     plt.show()
            #     plt.imshow(b[idd, :, :, 0])
            #     plt.show()
            #     plt.imshow(b[idd, :, :, 1])
            #     plt.show()
            #     plt.imshow(b[idd, :, :, 3])
            #     plt.show()

            summary_writer = tf.summary.FileWriter(flags.model_save_path, sess.graph)

            for step in range(int(global_step.eval()), TRAINING_STEPS):
                start_time = time.time()
                _, loss_value = sess.run([train_op, loss])
                duration = time.time() - start_time
                if step != 0 and step % 10 == 0:
                    num_examples_per_step = flags.batch_size * len(N_GPU)
                    examples_per_sec = num_examples_per_step / duration
                    sec_per_batch = duration / len(N_GPU)
                    format_str = '%s: step %d, loss = %.2f (%.1f examples/sec; %.3f sec/batch)'
                    print(format_str % (datetime.now(), step, loss_value, examples_per_sec, sec_per_batch))
                    if step >= 1000:
                        summary = sess.run(summary_op)
                        summary_writer.add_summary(summary, step)
                if step % 2000 == 0 or (step + 1) == TRAINING_STEPS:
                    checkpoint_path = os.path.join(flags.model_save_path, flags.model_name)
                    saver.save(sess, checkpoint_path, global_step=step)

            coord.request_stop()
            coord.join(threads)


def tower_loss(scope, images, of_stack, labels, flags):
    logits_spatial = z_inference.alex_inference_for_two_stream(images, flags.batch_size, 3, flags.num_class,
                                                               stream='_spatial')
    logits_temporal = z_inference.alex_inference_for_two_stream(of_stack, flags.batch_size, 2 * flags.of_stack_num,
                                                               flags.num_class, stream='_temporal')
    logits = (logits_spatial + logits_temporal)/2
    _ = get_loss(logits, labels)
    losses = tf.get_collection('losses', scope)
    total_loss = tf.add_n(losses, name='total_loss')
    return total_loss


def get_loss(logits, labels):
    labels = tf.cast(labels, tf.int64)
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=labels, logits=logits, name='cross_entropy_per_example')
    cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
    tf.add_to_collection('losses', cross_entropy_mean)
    return tf.add_n(tf.get_collection('losses'), name='total_loss')


# def get_loss(x, y_, regu, class_num, scope, reuse_variables=None):
#     # (input_batch, input_channel_num, regu, net_stream='spatial')
#     with tf.variable_scope(tf.get_variable_scope(), reuse=reuse_variables):
#         y = net_inference.inference(x, 3, class_num, regu, True, 'spatial')
#     cross_entropy = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=y_))
#     # regularization_loss = tf.add_n(tf.get_collection('spatial_loss', scope))
#     loss = cross_entropy  # + regularization_loss
#     return loss


def average_gradients(tower_grads):
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        grads = []
        for g, _ in grad_and_vars:
            expanded_g = tf.expand_dims(g, 0)
            grads.append(expanded_g)
        grad = tf.concat(grads, 0)
        grad = tf.reduce_mean(grad, 0)
        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
    return average_grads


if __name__ == '__main__':
    main()
    # tf.app.run()
