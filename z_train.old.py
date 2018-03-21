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

BATCH_SIZE = int(256 / 2)
LEARNING_RATE_BASE = 0.00003
LEARNING_RATE_DECAY = 0.99
REGULARIZATION_RATE = 0.0001
TRAINING_STEPS = 140000
MOVING_AVERAGE_DECAY = 0.99
DEMO = False
KEY = 'UCF101'
OF_STACK_NUMS = 5
split = 1
DATA_FRAMES_PATH = []
IN_NET_SIZE = [224, 224]

if platform.system() == 'Windows':
    N_GPU = [0]
    DATA_FRAMES_PATH = r'D:\Desktop\DEMO_FILE\UCF101pic'
    DATA_OF_PATH = DATA_FRAMES_PATH + '_of'
    DATA_OF_DIC_PATH = DATA_OF_PATH + '_dict'
    MODEL_SAVE_PATH = r"D:\Desktop\Deep_Learning_with_TensorFlow\models_z"
    MODEL_NAME = "model.ckpt"
elif platform.system() == 'Linux':
    N_GPU = [0, 1, 2, 3]
    DATA_FRAMES_PATH = '/home/zbh/Desktop/alpha_2_zbh/UCF101pic'
    DATA_OF_PATH = DATA_FRAMES_PATH + '_of'
    DATA_OF_DIC_PATH = DATA_OF_PATH + '_dict'
    MODEL_SAVE_PATH = r"/home/zbh/Desktop/alpha_1_zbh/models"
    MODEL_NAME = "model.ckpt"
else:
    exit('ZBH: unknown platform system %s.' % platform.system())
ROOT_PATH = os.path.split(DATA_FRAMES_PATH)[0]
DATA_TF_FILE_PATH = os.path.join(ROOT_PATH, '%s_tfrecords_%sof' % (KEY, OF_STACK_NUMS))
if KEY == 'UCF101':
    FILE_LIST_PATH = os.path.join(ROOT_PATH, 'ucfTrainTestlist')


def get_2stream_input(tfrecord_list, for_eval=False):
    reader = tf.TFRecordReader()
    filename_queue = tf.train.string_input_producer(tfrecord_list, shuffle=not for_eval)  # , num_epochs=1)
    # filename_queue = tf.train.string_input_producer(
    #    [r'D:\Desktop\DEMO_FILE\UCF101_tfrecords_10of\data.tfrecords-v_ApplyEyeMakeup_g02_c04'])
    _, serialized_example = reader.read(filename_queue)
    read_tfrecord = dr.make_feature_eval_str(OF_STACK_NUMS)
    features = eval(read_tfrecord)
    label = features['label']
    # decode image
    image = tf.image.decode_jpeg(features['image_raw'])
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    image = tf.image.resize_images(image, IN_NET_SIZE, method=0)
    image = tf.reshape(image, IN_NET_SIZE + [3])
    # decode of_stack
    of_min_max_arr = tf.decode_raw(features['of_min_max_raw'], tf.float32)
    of_stack_arr = []
    for n in range(OF_STACK_NUMS):
        of_ext = ['_x', '_y']
        for xy in range(2):
            of_name = 'of_' + str(n) + of_ext[xy]
            of_jpg = tf.cast(tf.image.decode_jpeg(features[of_name]), tf.float32)
            of_min = of_min_max_arr[2 * (n + xy)]
            of_max = of_min_max_arr[2 * (n + xy) + 1]
            of_org = of_jpg * (of_max - of_min) / 255. + of_min
            if n + xy == 0:
                of_stack_arr = of_org
            else:
                of_stack_arr = tf.concat([of_stack_arr, of_org], 2)
    of_stack_arr = tf.image.resize_images(of_stack_arr, IN_NET_SIZE, method=0)
    of_stack_arr = tf.reshape(of_stack_arr, IN_NET_SIZE + [2 * OF_STACK_NUMS])
    # set batch
    if for_eval is False:
        min_after_dequeue = 1000
        batch_size = BATCH_SIZE
        capacity = min_after_dequeue + 3 * batch_size
        image_batch, of_stack_batch, label_batch = tf.train.shuffle_batch([image, of_stack_arr, label],
                                                                          batch_size=batch_size, capacity=capacity,
                                                                          num_threads=cpu_count(),
                                                                          min_after_dequeue=min_after_dequeue)
        return image_batch, of_stack_batch, label_batch
    else:
        # decode frame info
        rootb = features['root']
        nameb = features['name']
        idexb = features['idex']
        min_after_dequeue = 1000
        batch_size = BATCH_SIZE
        capacity = min_after_dequeue + 3 * batch_size
        image_batch, of_stack_batch, label_batch, root_batch, name_batch, index_batch = tf.train.batch(
            [image, of_stack_arr, label, rootb, nameb, idexb],
            batch_size=batch_size, capacity=capacity,
            num_threads=cpu_count())
        return image_batch, of_stack_batch, label_batch, root_batch, name_batch, index_batch


def get_loss(x, y_, regu, class_num, scope, reuse_variables=None):
    # (input_batch, input_channel_num, regu, net_stream='spatial')
    with tf.variable_scope(tf.get_variable_scope(), reuse=reuse_variables):
        y = net_inference.inference(x, 3, class_num, regu, True, 'spatial')
    cross_entropy = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=y_))
    #regularization_loss = tf.add_n(tf.get_collection('spatial_loss', scope))
    loss = cross_entropy# + regularization_loss
    return loss


def main():
    dr.check_or_create_path([DATA_TF_FILE_PATH])
    [class_list_init, train_list_init, test_list_init] = dr.get_train_and_test_list(KEY, FILE_LIST_PATH, split)
    [class_list, train_list, test_list] = dr.if_make_demo_list(class_list_init, train_list_init, test_list_init, DEMO)
    # [['2', 'ApplyLipstick'],...],  [['YoYo/v_YoYo_g25_c05.avi', '101'],...],  [['YoYo/v_YoYo_g25_c05.avi'],...]
    dr.naive_check_multiprocessing(class_list, DATA_FRAMES_PATH, DATA_OF_PATH, DATA_OF_DIC_PATH)
    dr.check_or_create_tfrecords_multiprocessing(train_list + test_list, class_list, DATA_FRAMES_PATH,
                                                 DATA_TF_FILE_PATH, OF_STACK_NUMS, )
    tfrecord_list = dr.create_tfrecords_list(train_list, DATA_TF_FILE_PATH)

    with tf.Graph().as_default(), tf.device('/cpu:0'):
        image_batch, of_stack_batch, label_batch = get_2stream_input(tfrecord_list)

        #regu = tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)
        regu = None
        global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False)
        learning_rate = tf.train.exponential_decay(LEARNING_RATE_BASE, global_step,
                                                   TRAINING_STEPS / math.log(0.005, LEARNING_RATE_DECAY),
                                                   LEARNING_RATE_DECAY)
        opt = tf.train.GradientDescentOptimizer(learning_rate)
        tower_grads = []
        reuse_variables = False
        for i in N_GPU:
            with tf.device('/gpu:%d' % i):
                with tf.name_scope('GPU_%d' % i) as scope:
                    cur_loss = get_loss(image_batch, label_batch, regu, len(class_list), scope, reuse_variables)
                    reuse_variables = True
                    # tf.get_variable_scope().reuse_variables()
                    grads = opt.compute_gradients(cur_loss)
                    tower_grads.append(grads)

        grads = average_gradients(tower_grads)
        for grad, var in grads:
            if grad is not None:
                tf.summary.histogram('gradients_on_average/%s' % var.op.name, grad)

        apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)
        for var in tf.trainable_variables():
            tf.summary.histogram(var.op.name, var)

        variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
        variables_to_average = (tf.trainable_variables() + tf.moving_average_variables())
        variables_averages_op = variable_averages.apply(variables_to_average)

        train_op = tf.group(apply_gradient_op, variables_averages_op)
        saver = tf.train.Saver(tf.all_variables())
        summary_op = tf.summary.merge_all()

        init = (tf.global_variables_initializer(), tf.local_variables_initializer())
        gpu_options = tf.GPUOptions(allow_growth=True)
        config = tf.ConfigProto(allow_soft_placement=True, gpu_options=gpu_options)
        with tf.Session(config=config) as sess:
            sess.run(init)
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)

            summary_writer = tf.summary.FileWriter(MODEL_SAVE_PATH, sess.graph)

            for step in range(TRAINING_STEPS):
                start_time = time.time()
                _, loss_value = sess.run([train_op, cur_loss])
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
