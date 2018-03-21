def check_or_create_tfrecords(a_list, class_list, save_path, stack_num,
                              tf_base='data.tfrecords-'):  # save_path = DATA_TF_FILE_PATH
    for file in a_list:  # ['ApplyLipstick/v_ApplyLipstick_g25_c04.avi', '2']
        [root, name] = os.path.splitext(file[0])[0].split('/')  # ['ApplyLipstick','v_ApplyLipstick_g25_c04']
        tf_name = os.path.join(save_path, (tf_base + name))  # save_path\data.tfrecords-v_ApplyLipstick_g25_c04
        if not os.path.isfile(tf_name):
            label = [x[1] for x in class_list].index(root)  # [['1', 'ApplyEyeMakeup'], ['2', 'ApplyLipstick']] > 1
            pic_path = os.path.join(DATA_FRAMES_PATH, root, name)  # UCF101pic\ApplyLipstick\v_ApplyLipstick_g25_c04
            of_path = os.path.join(DATA_OF_PATH, root, name)  # UCF101pic_of\ApplyLipstick\v_ApplyLipstick_g25_c04
            writer = tf.python_io.TFRecordWriter(tf_name)
            order = sorted(os.listdir(pic_path),
                           key=lambda x: int(x.split('.')[0]))  # ['1.jpg','2.jpg',...,'198.jpg']
            for j in order[:-stack_num]:
                pic_file = os.path.join(pic_path, j)
                image_raw_data = tf.gfile.FastGFile(pic_file, 'rb').read()
                # image_raw = tf.image.decode_jpeg(image_raw_data).eval().tostring()

                    feature={'label': _int64_feature(label), 'image_raw': _bytes_feature(image_raw_data)}))
                writer.write(example.SerializeToString())
            print('%d,bytes.' % sys.getsizeof(image_raw_data))
            # print('%d,bytes.' % sys.getsizeof(image_raw))
            writer.close()
            print('%s have just been written.' % tf_name)
        else:
            print('%s already exists.' % tf_name)

def check_or_create_tfrecords(a_list, class_list, check_path, save_path, stack_num, tf_base='list.tfrecords-'):
    for file in a_list:  # ['YoYo/v_YoYo_g25_c04.avi', '2'] \ ['YoYo/v_YoYo_g25_c04.avi']
        [root, name] = os.path.splitext(file[0])[0].split('/')  # ['ApplyLipstick','v_ApplyLipstick_g25_c04']
        tf_name = os.path.join(save_path, (tf_base + name))  # save_path\data.tfrecords-v_ApplyLipstick_g25_c04
        if os.path.isfile(tf_name) and os.path.getsize(tf_name) == 0:
            os.remove(tf_name)
        if not os.path.isfile(tf_name):
            label = [x[1] for x in class_list].index(
                root)  # [['1', 'ApplyEyeMakeup'], ['2', 'ApplyLipstick']] > 1
            pic_path = os.path.join(check_path, root, name)  # UCF101pic\ApplyLipstick\v_ApplyLipstick_g25_c04
            # of_path = os.path.join(DATA_OF_PATH, root, name)  # UCF101pic_of\ApplyLipstick\v_ApplyLipstick_g25_c04
            writer = tf.python_io.TFRecordWriter(tf_name)
            order = sorted(os.listdir(pic_path),
                           key=lambda x: int(x.split('.')[0]))  # ['1.jpg','2.jpg',...,'198.jpg']
            for j in order[:-stack_num]:
                # pic_file = os.path.join(pic_path, jpg)
                # image_raw_data = tf.gfile.FastGFile(pic_file, 'rb').read()
                # image_raw = tf.image.decode_jpeg(image_raw_data).eval().tostring()
                idex = os.path.splitext(j)[0]
                example = tf.train.Example(features=tf.train.Features(
                    feature={'root': _bytes_feature(bytes(root, encoding='utf-8')),
                             'name': _bytes_feature(bytes(name, encoding='utf-8')),
                             'idex': _bytes_feature(bytes(idex, encoding='utf-8')),
                             'label': _int64_feature(label)
                             }))
                writer.write(example.SerializeToString())
                # print('%d,bytes.' % sys.getsizeof(example))
                # print('%d,bytes.' % sys.getsizeof(image_raw))
            writer.close()
            print('%s has just been written.' % tf_name)
        else:
            print('%s already exists.' % tf_name)
        if os.path.getsize(tf_name) == 0:
            exit('ZBH: something wrong at %s' % tf_name)


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

  # threads = []
    # for em in split_list:
    #     threads.append(threading.Thread(target=check_or_create_tfrecords,
    #                                     args=(em, class_list, check_path, save_path, stack_num, tf_base)))
    # for t in threads:
    #     t.setDaemon(True)
    #     t.start()
    # t.join()

def divide_list(a_list, num):
    if len(a_list) < num:
        exit('ZBH: list length is less than threading number: %d.' % num)
    # split_list = [[] * 1] * num NO : COPY APPEND IN EVERY ELEMENT
    split_list = [[] for i in range(num)]
    # split_list = []
    # for i in range(num):
    #     split_list.append([])
    for j in range(len(a_list)):
        idx = j % num
        split_list[idx].append(a_list[j])
    return split_list