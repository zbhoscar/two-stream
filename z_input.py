import os
import tensorflow as tf
import datasets_reader as dr
import numpy as np
from multiprocessing import cpu_count
import time


def distort_color(image, color_ordering=0):
    if color_ordering == 0:
        image = tf.image.random_brightness(image, 32. / 255.)
        image = tf.image.random_saturation(image, 0.5, 1.5)
        image = tf.image.random_hue(image, 0.2)
        image = tf.image.random_contrast(image, 0.5, 1.5)
    else:
        image = tf.image.random_saturation(image, 0.5, 1.5)
        image = tf.image.random_brightness(image, 32. / 255.)
        image = tf.image.random_hue(image, 0.2)
        image = tf.image.random_contrast(image, 0.5, 1.5)
    return tf.clip_by_value(image, 0.0, 1.0)


def get_2stream_input(flags, of_stack_num, crop_size=224):
    time_tab = time.strftime("%Y%m%d%H%M", time.localtime())
    data_parent_path = os.path.dirname(flags.root_path)
    data_frames_path = flags.root_path                                                   # ### VIDEO FRAME FOLDER ###
    data_of_path = data_frames_path + '_of'
    data_of_dic_path = data_of_path + '_dict'
    data_tf_file_path = '%s_tfrecords_%sof' % (data_frames_path, of_stack_num)
    if flags.model_save_path == '':
        flags.model_save_path = os.path.join(flags.root_path, flags.cousin_path, flags.key + '_' + time_tab)
    file_list_path = ''
    if flags.key == 'UCF101':
        file_list_path = os.path.join(data_parent_path, 'ucfTrainTestlist')
    print(' Datasets: %s.\tdata_parent_path is %s\n' % (flags.key, data_parent_path),
          'data_frames_path  is %s\n' % data_frames_path, 'data_of_path      is %s\n' % data_of_path,
          'data_of_dic_path  is %s\n' % data_of_dic_path, 'data_tf_file_path is %s\n' % data_tf_file_path,
          'model_save_path   is %s\n' % flags.model_save_path, 'model_name        is %s' % flags.model_name)

    # check frame/of/dict files
    dr.check_or_create_path([data_tf_file_path])
    [class_list_init, train_list_init, test_list_init] = dr.get_train_and_test_list(flags.key, file_list_path, split=1)
    [class_list, train_list, test_list] = dr.if_make_demo_list(class_list_init, train_list_init, test_list_init,
                                                               flags.demo)
    flags.num_class = len(class_list)
    # [['2', 'ApplyLipstick'],...],  [['YoYo/v_YoYo_g25_c05.avi', '101'],...],  [['YoYo/v_YoYo_g25_c05.avi'],...]
    dr.naive_check_multiprocessing(class_list, data_frames_path, data_of_path, data_of_dic_path)
    dr.check_or_create_tfrecords_multiprocessing(train_list + test_list, class_list, data_frames_path,
                                                 data_tf_file_path, of_stack_num)
    tfrecord_list = dr.create_tfrecords_list(train_list, data_tf_file_path)

    # get input from tfrecords_list
    reader = tf.TFRecordReader()
    filename_queue = tf.train.string_input_producer(tfrecord_list, shuffle=not flags.bool_eval)  # , num_epochs=1)
    # filename_queue = tf.train.string_input_producer(
    #    [r'D:\Desktop\DEMO_FILE\UCF101_tfrecords_10of\data.tfrecords-v_ApplyEyeMakeup_g02_c04'])
    _, serialized_example = reader.read(filename_queue)
    read_tfrecord = dr.make_feature_eval_str(of_stack_num)  # commend str according to of_stack_num
    features = eval(read_tfrecord)
    label = features['label']
    # decode image
    image = tf.image.decode_jpeg(features['image_raw'])
    if image.dtype != tf.float32:
        image = tf.image.convert_image_dtype(image, dtype=tf.float32)

    # PRE-PROCESSING frame images
    # image = tf.image.resize_images(image, in_net_size, method=0)
    image = tf.image.resize_image_with_crop_or_pad(image, crop_size, crop_size)
    # image = tf.reshape(image, [crop_size, crop_size, 3])
    image = tf.image.random_flip_left_right(image)
    image = distort_color(image, np.random.randint(2))

    # decode of_stack
    of_min_max_arr = tf.decode_raw(features['of_min_max_raw'], tf.float32)
    of_stack_arr = []
    for n in range(of_stack_num):
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
    of_stack_arr = tf.image.resize_image_with_crop_or_pad(of_stack_arr, crop_size, crop_size)
    # of_stack_arr = tf.image.resize_images(of_stack_arr, in_net_size, method=0)
    of_stack_arr = tf.reshape(of_stack_arr, [crop_size, crop_size, 2 * of_stack_num])
    # decode frame info
    rootb = features['root']
    nameb = features['name']
    idexb = features['idex']
    min_after_dequeue = 1000
    capacity = min_after_dequeue + 3 * flags.batch_size
    # get batch
    if flags.bool_eval is False:
        [image_batch, of_stack_batch, label_batch, root_batch, name_batch, index_batch] = tf.train.shuffle_batch(
            [image, of_stack_arr, label, rootb, nameb, idexb],
            batch_size=flags.batch_size,
            capacity=capacity,
            num_threads=int(cpu_count()),
            min_after_dequeue=min_after_dequeue)
    else:
        [image_batch, of_stack_batch, label_batch, root_batch, name_batch, index_batch] = tf.train.batch(
            [image, of_stack_arr, label, rootb, nameb, idexb],
            batch_size=flags.batch_size, capacity=capacity,
            num_threads=int(cpu_count()))
    return image_batch, of_stack_batch, label_batch, root_batch, name_batch, index_batch, flags
