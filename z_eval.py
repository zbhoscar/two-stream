# -*- coding: UTF-8 -*- #
import platform
import tensorflow as tf
import os
import datasets_reader as dr
import matplotlib.pyplot as plt
from multiprocessing import cpu_count
import alex2 as netz
import time
from datetime import datetime
import numpy as np

BATCH_SIZE = 64  # int(256 / 2)
LEARNING_RATE_BASE = 0.000001
LEARNING_RATE_DECAY = 0.99
REGULARIZATION_RATE = 0.0001
TRAINING_STEPS = 70000
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
    MODEL_SAVE_PATH = r"D:\Desktop\Deep_Learning_with_TensorFlow\models"
    MODEL_NAME = "model.ckpt"
    DEMO = True
elif platform.system() == 'Linux':
    N_GPU = [0, 1, 2, 3]
    DATA_FRAMES_PATH = '/home/zbh/Desktop/alpha_2_zbh/UCF101pic'
    DATA_OF_PATH = DATA_FRAMES_PATH + '_of'
    DATA_OF_DIC_PATH = DATA_OF_PATH + '_dict'
    MODEL_SAVE_PATH = r"/home/zbh/Desktop/Deep_Learning_with_TensorFlow/models"
    MODEL_NAME = "model.ckpt"
else:
    exit('ZBH: unknown platform system %s.' % platform.system())
ROOT_PATH = os.path.split(DATA_FRAMES_PATH)[0]
DATA_TF_FILE_PATH = os.path.join(ROOT_PATH, '%s_tfrecords_%sof' % (KEY, OF_STACK_NUMS))
if KEY == 'UCF101':
    FILE_LIST_PATH = os.path.join(ROOT_PATH, 'ucfTrainTestlist')


def get_2stream_input(tfrecord_list, for_eval=False):
    reader = tf.TFRecordReader()
    filename_queue = tf.train.string_input_producer(tfrecord_list, shuffle=True)#not for_eval)  # , num_epochs=1)
    # filename_queue = tf.train.string_input_producer(
    #    [r'D:\Desktop\DEMO_FILE\UCF101_tfrecords_10of\data.tfrecords-v_ApplyEyeMakeup_g02_c04'])
    _, serialized_example = reader.read(filename_queue)
    read_tfrecord = dr.make_feature_eval_str(OF_STACK_NUMS)
    features = eval(read_tfrecord)
    # decode frame info
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

    if for_eval == False:
        # set batch
        min_after_dequeue = 1000
        batch_size = BATCH_SIZE
        capacity = min_after_dequeue + 3 * batch_size

        image_batch, of_stack_batch, label_batch = tf.train.shuffle_batch([image, of_stack_arr, label],
                                                                          batch_size=batch_size, capacity=capacity,
                                                                          num_threads=cpu_count(),
                                                                          min_after_dequeue=min_after_dequeue)
        return image_batch, of_stack_batch, label_batch
    else:
        rootb = features['root']
        nameb = features['name']
        idexb = features['idex']

        min_after_dequeue = 1000
        batch_size = BATCH_SIZE * 2
        capacity = min_after_dequeue + 3 * batch_size
        image_batch, of_stack_batch, label_batch, root_batch, name_batch, index_batch = tf.train.batch(
            [image, of_stack_arr, label, rootb, nameb, idexb], batch_size=batch_size, capacity=capacity,
            num_threads=int(cpu_count() / 2))

        return image_batch, of_stack_batch, label_batch, root_batch, name_batch, index_batch


def main():
    dr.check_or_create_path([DATA_TF_FILE_PATH])
    [class_list_init, train_list_init, test_list_init] = dr.get_train_and_test_list(KEY, FILE_LIST_PATH, split)
    [class_list, train_list, test_list] = dr.if_make_demo_list(class_list_init, train_list_init, test_list_init, DEMO)
    # [['2', 'ApplyLipstick'],...],  [['YoYo/v_YoYo_g25_c05.avi', '101'],...],  [['YoYo/v_YoYo_g25_c05.avi'],...]
    dr.naive_check_multiprocessing(class_list, DATA_FRAMES_PATH, DATA_OF_PATH, DATA_OF_DIC_PATH)
    dr.check_or_create_tfrecords_multiprocessing(train_list + test_list, class_list, DATA_FRAMES_PATH,
                                                 DATA_TF_FILE_PATH, OF_STACK_NUMS, )
    tfrecord_test_list = dr.create_tfrecords_list(test_list, DATA_TF_FILE_PATH)

    with tf.Graph().as_default() as g:
        image_batch, of_stack_batch, label_batch_, root_batch, name_batch, index_batch = get_2stream_input(
            tfrecord_test_list, for_eval=True)
        with tf.variable_scope(tf.get_variable_scope()):
            label_batch = netz.inference(image_batch, 3, len(class_list), None, net_stream='spatial')
        # correct_prediction=tf.equal(tf.argmax(label_batch,1), label_batch_)
        # accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))   
        variable_averages=tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY)
        variables_to_restore=variable_averages.variables_to_restore()
        saver=tf.train.Saver(variables_to_restore)






        #init = (tf.global_variables_initializer(), tf.local_variables_initializer())
        gpu_options = tf.GPUOptions(allow_growth=True)
        config = tf.ConfigProto(allow_soft_placement=True, gpu_options=gpu_options)
        with tf.Session(config=config) as sess:
            #sess.run(init)
            ckpt = tf.train.get_checkpoint_state(MODEL_SAVE_PATH)
            saver.restore(sess,ckpt.model_checkpoint_path)
            global_step=ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
            print(global_step)

            coord_eval = tf.train.Coordinator()
            threads_eval = tf.train.start_queue_runners(sess=sess, coord=coord_eval)

            for i in range(200):
                a, b =sess.run([label_batch,label_batch_])
                print('************************************')
                print(a[0],a.shape)
                print(b,b.shape)

            # whole = []
            # label_all_ground = []
            # label_all_eval = []
            # batch_go_on = True
            # while batch_go_on == True:
            #     aa, bb, cc, dd, ee, ff = sess.run([root_batch, name_batch, index_batch, 
            #                                     tf.argmax(label_batch,1), label_batch_,accuracy])
            #     temp = list(zip(aa, bb, cc))
            #     whole = whole + temp

            #     label_all_ground = label_all_ground + list(ee)
            #     label_all_eval = label_all_eval + list(dd)



            #     if whole[0] in whole[-BATCH_SIZE:] and len(whole) > BATCH_SIZE:
            #         whole_end = whole.index(whole[0], -BATCH_SIZE)
            #         whole = whole[:whole_end]
            #         label_all_ground = label_all_ground[:whole_end]
            #         label_all_eval = label_all_eval[:whole_end]
            #         batch_go_on = False

            #     print('**************************')
            #     print(dd)
            #     print(ee)
            #     print(ff,np.mean(np.int16(np.equal(label_all_eval, label_all_ground))))

            coord_eval.request_stop()
            coord_eval.join(threads_eval)

            # print(step,len(whole))
            # print(whole[0:5])
            # print(exam[0:5])
            # print(whole[-5:])
            # print(exam[-5:])



if __name__ == '__main__':
    main()
