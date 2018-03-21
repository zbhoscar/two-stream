# -*- coding: UTF-8 -*- #
import os
import tensorflow as tf
import multiprocessing
import numpy as np
from PIL import Image


def txt_read_in_lines(filepath, split=' '):
    arr = []
    file = open(filepath)
    contents = file.readlines()
    for item in contents:
        content = item.strip()
        temp = content.split(split)
        arr.append(temp)
    return arr


def get_train_and_test_list(key, filepath, split=1):
    class_list = []
    train_list = []
    test_list = []
    if key == 'UCF101':
        class_list = txt_read_in_lines(os.path.join(filepath, 'classInd.txt'))
        if split == 1:
            train_list = txt_read_in_lines(os.path.join(filepath, 'trainlist01.txt'))
            test_list = txt_read_in_lines(os.path.join(filepath, 'testlist01.txt'))
        elif split == 2:
            train_list = txt_read_in_lines(os.path.join(filepath, 'trainlist02.txt'))
            test_list = txt_read_in_lines(os.path.join(filepath, 'testlist02.txt'))
        elif split == 3:
            train_list = txt_read_in_lines(os.path.join(filepath, 'trainlist03.txt'))
            test_list = txt_read_in_lines(os.path.join(filepath, 'testlist03.txt'))
        else:
            exit('ZBH: wrong split num!')
    return class_list, train_list, test_list


def if_make_demo_list(class_list, train_list, test_list, demo=False):
    if demo is True:
        class_demo_list = class_list[0:2]
        i = 0
        while class_list[2][1] not in train_list[i][0]:
            i = i + 1
        train_demo_list = train_list[0:i]
        j = 0
        while class_list[2][1] not in test_list[j][0]:
            j = j + 1
        test_demo_list = test_list[0:j]
        return class_demo_list, train_demo_list, test_demo_list
    elif demo is False:
        return class_list, train_list, test_list
        # [['2', 'ApplyLipstick'],...],  [['YoYo/v_YoYo_g25_c05.avi', '101']...],  [['YoYo/v_YoYo_g25_c05.avi',...]


def check_or_create_path(path_list, create=True):
    for a_path in path_list:
        if not os.path.exists(a_path) and create:
            os.makedirs(a_path)
        elif not os.path.exists(a_path) and not create:
            exit('ZBH: %s does not exist.' % a_path)


def naive_check(class_list, img_path, of_path, dict_path):
    dict_ok = True
    for i in class_list:
        clas = i[1]  # 'ApplyEyeMakeup'
        print('%s' % clas, end='.. ')
        im_sub_path = os.path.join(img_path, clas)  # 'D:\\Desktop\\DEMO_FILE\\UCF101pic_256\\ApplyEyeMakeup'
        of_sub_path = os.path.join(of_path, clas)  # 'D:\\Desktop\\DEMO_FILE\\UCF101pic_256_of\\ApplyEyeMakeup'
        check_or_create_path([im_sub_path, of_sub_path], create=False)
        sub_path = os.listdir(im_sub_path)  # ['v_ApplyEyeMakeup_g01_c01', 'v_ApplyEyeMakeup_g01_c02',.]
        if len(sub_path) != len(os.listdir(of_sub_path)):  # if im_sub_path and of_sub_path have same folders
            exit('ZBH: sub_path does not equal between \n\t%s \t\tand \n\t%s.' % (im_sub_path, of_sub_path))
        of_dict_file = os.path.join(dict_path, clas + '.txt')  # '...\\UCF101pic_256_of_dict\\ApplyEyeMakeup.txt'
        of_dict = of_dict_reader(of_dict_file)  # read ApplyEyeMakeup.txt
        for j in sub_path:
            im_sample_path = os.path.join(im_sub_path, j)  # '...pic_256\\ApplyEyeMakeup\\v_ApplyEyeMakeup_g01_c02'
            of_sample_path = os.path.join(of_sub_path, j)  # '...pic_256_of\\ApplyEyeMakeup\\v_ApplyEyeMakeup_g01_c02'
            im_list = sorted(os.listdir(im_sample_path), key=lambda x: int(x[:-4]))  # ['1.jpg', '2.jpg', '3.jpg', ...
            of_list = sorted(os.listdir(of_sample_path), key=lambda x: int(x[:-6]))  # ['1_x.jpg', '1_y.jpg', '2_x.jpg',
            im_num = len(im_list)  # 122
            of_num = len(of_list)  # 242
            # check order in im_list and of_list
            if 2 * im_num != of_num + 2 or im_list[0] != '1.jpg' or of_list[0] != '1_x.jpg' or im_list[-1] != (
                    '%d.jpg' % im_num) or of_list[-1] != ('%d_y.jpg' % (im_num - 1)):
                exit('ZBH: something wrong at %s.\n\tkey values: %d, %d, %s, %s, %s, %s' % (
                    j, im_num, of_num, im_list[0], of_list[0], im_list[-1], of_list[-1]))
            for k in of_list:  # '1_x.jpg' in  ['1_x.jpg', '1_y.jpg', '2_x.jpg',...]
                if j + '/' + k not in of_dict:  # 'v_ApplyEyeMakeup_g01_c02/1_x.jpg' not in of_dict?
                    of_jpg_path = os.path.join(of_sample_path, k)
                    of_jpg = np.array(Image.open(of_jpg_path))  # of_jpg is unit8 pic
                    # if not 0, then should in dict:
                    if not np.max(of_jpg) + np.min(of_jpg) == 0 and not np.max(of_jpg) * np.min(of_jpg) == 0:
                        print('ZBH: something wrong in %s' % of_jpg_path)
                        dict_ok = False
    return dict_ok


def divide_list(a_list, num):
    split_list = [[] for i in range(num)]
    for j in range(len(a_list)):
        idx = j % num
        split_list[idx].append(a_list[j])
    return split_list


# def get_reshaped_gfile(pic_path, j, min_side=256):
#     pic_file = os.path.join(pic_path, j)  # UCF101pic\ApplyLipstick\v_ApplyLipstick_g25_c04\1.jpg
#     # resize images to min = 256
#     img = Image.open(pic_file)
#     new_size = [round(min_side * x / min(img.size)) for x in img.size]  # new_size for img_min_side=256
#     reshaped_img = img.resize(new_size)
#     reshaped_pic_file = os.path.join(pic_path, 'tmp_' + j)
#     reshaped_img.save(reshaped_pic_file, 'jpeg')
#     image_raw = tf.gfile.FastGFile(reshaped_pic_file, 'rb').read()
#     os.remove(reshaped_pic_file)
#     return image_raw
def get_reshaped_gfile(pic_path, j, min_side=256):
    pic_file = os.path.join(pic_path, j)  # UCF101pic\ApplyLipstick\v_ApplyLipstick_g25_c04\1.jpg
    # resize images to min = 256
    # img = Image.open(pic_file)
    image_raw = tf.gfile.FastGFile(pic_file, 'rb').read()
    image_orig = tf.image.decode_jpeg(image_raw)
    orig_size = image_orig.eval().shape[:2]
    new_size = [round(min_side * x / min(orig_size)) for x in orig_size]  # new_size for img_min_side=256#
    image = tf.image.convert_image_dtype(image_orig, dtype=tf.float32)
    image = tf.image.resize_images(image, new_size, method=0)
    image = tf.image.convert_image_dtype(image, dtype=tf.uint8)
    encoded_image = tf.image.encode_jpeg(image, quality=50, optimize_size=True)
    output = encoded_image.eval()
    with tf.gfile.GFile(r'D:\Desktop\test.jpg', 'wb') as f:
        f.write(output)
    return encoded_image.eval()


def check_or_create_tfrecords(a_list, class_list, check_path, save_path, stack_num, tf_base='list.tfrecords-'):
    for file in a_list:  # ['YoYo/v_YoYo_g25_c04.avi', '2'] \ ['YoYo/v_YoYo_g25_c04.avi']
        [root, name] = os.path.splitext(file[0])[0].split('/')  # ['ApplyLipstick','v_ApplyLipstick_g25_c04']
        tf_name = os.path.join(save_path, (tf_base + name))  # save_path\data.tfrecords-v_ApplyLipstick_g25_c04
        if os.path.isfile(tf_name) and os.path.getsize(tf_name) == 0:  # clean wrong tfrecord: exist but 0 bytes
            os.remove(tf_name)
        if not os.path.isfile(tf_name):  # if not （exist) file
            label = [x[1] for x in class_list].index(root)  # [['1', 'ApplyEyeMakeup'], ['2', 'ApplyLipstick']] > 0
            # check_path = r'D:\Desktop\DEMO_FILE\UCF101pic';root='ApplyLipstick';name='v_ApplyLipstick_g25_c04'
            pic_path = os.path.join(check_path, root, name)  # UCF101pic\ApplyLipstick\v_ApplyLipstick_g25_c04
            of_path = os.path.join(check_path + '_of', root, name)  # UCF101pic_of\ApplyLipstick\v_ApplyLipstick_g25_c04
            writer = tf.python_io.TFRecordWriter(tf_name)
            order = sorted(os.listdir(pic_path), key=lambda x: int(x.split('.')[0]))  # ['1.jpg','2.jpg',...,'198.jpg']
            of_dict_file = os.path.join(check_path + '_of_dict', root + '.txt')
            of_dict = of_dict_reader(of_dict_file)
            for j in order[:-stack_num]:  # '1.jpg'
                # image_raw = get_reshaped_gfile(pic_path, j, min_side=256)             # *****
                pic_file = os.path.join(pic_path, j)
                image_raw = tf.gfile.FastGFile(pic_file, 'rb').read()
                idex = os.path.splitext(j)[0]
                of_bytes_list = [[[] for xy in range(2)] for k in range(stack_num)]
                of_min_max_arr = np.empty([stack_num, 2, 2], dtype='float32')
                of_stack_eval_str = ''
                for n in range(stack_num):
                    of_idx = int(idex) + n
                    of_ext = ['_x', '_y']
                    of_pic_path_base = os.path.join(of_path, str(of_idx))  # .../v_ApplyEyeMakeup_g01_c01/1
                    for xy in range(2):
                        of_raw_from_file = of_pic_path_base + of_ext[xy] + '.jpg'  # '.../1_x.jpg'
                        of_bytes_list[n][xy] = tf.gfile.FastGFile(of_raw_from_file, 'rb').read()
                        of_name = 'of_' + str(n) + of_ext[xy]  # 'of_0_x'
                        of_astr = 'of_bytes_list[%d][%d]' % (n, xy)  # 'of_bytes_list[0][0]'
                        of_eval_str = """'%s': _bytes_feature(%s),""" % (of_name, of_astr)
                        # "'of_0_x': _bytes_feature(of_bytes_list[0][0]),"
                        of_stack_eval_str = of_stack_eval_str + of_eval_str
                        min_max_ext = ['min', 'max']
                        word = name + '/' + str(of_idx) + of_ext[xy] + '.jpg'
                        for min_max in range(2):
                            # print(of_dict_file, n, xy, min_max, word, min_max_ext[min_max])
                            if word in of_dict:
                                of_min_max_arr[n][xy][min_max] = of_dict[word][min_max_ext[min_max]]
                            else:
                                of_min_max_arr[n][xy][min_max] = min_max  # of_dict[word][min_max_ext[min_max]]
                of_min_max_raw = of_min_max_arr.tostring()
                write_tfrecord_base = """tf.train.Example(features=tf.train.Features(
                    feature={'root': _bytes_feature(bytes(root, encoding='utf-8')),
                             'name': _bytes_feature(bytes(name, encoding='utf-8')),
                             'idex': _bytes_feature(bytes(idex, encoding='utf-8')),
                             'label': _int64_feature(label),
                             'image_raw': _bytes_feature(image_raw),
                             'of_min_max_raw': _bytes_feature(of_min_max_raw),
                             }))"""
                write_tfrecord = write_tfrecord_base[:-3] + of_stack_eval_str + '}))'
                # example = tf.train.Example(features=tf.train.Features(
                #     feature={'root': _bytes_feature(bytes(root, encoding='utf-8')),
                #              'name': _bytes_feature(bytes(name, encoding='utf-8')),
                #              'idex': _bytes_feature(bytes(idex, encoding='utf-8')),
                #              'label': _int64_feature(label)}))
                example = eval(write_tfrecord)
                writer.write(example.SerializeToString())
            writer.close()
            print('%s has just been written.' % tf_name)
        else:
            # print('%s already exists.' % tf_name)
            if np.random.random(1) < 0.1:
                print('>', end='')
        if os.path.getsize(tf_name) == 0:
            exit('\nZBH: something wrong at %s' % tf_name)


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def check_or_create_tfrecords_multiprocessing(a_list, class_list, check_path, save_path, stack_num,
                                              tf_base='data.tfrecords-', num=int(multiprocessing.cpu_count())):
    ext_num = min(len(a_list), num)
    split_list = divide_list(a_list, ext_num)

    # config = tf.ConfigProto()
    # config.gpu_options.allow_growth = True
    # config.log_device_placement = True
    # with tf.Session(config=config):
    print('Checking tfrecords:', end='\n    ')
    p = multiprocessing.Pool(ext_num)
    for em in split_list:
        p.apply_async(check_or_create_tfrecords, args=(em, class_list, check_path, save_path, stack_num, tf_base))
    p.close()
    p.join()
    print(' ALL tfrecords fine!')


def naive_check_multiprocessing(a_list, img_path, of_path, dict_path, num=int(multiprocessing.cpu_count())):
    ext_num = min(len(a_list), num)
    split_list = divide_list(a_list, ext_num)
    p = multiprocessing.Pool(ext_num)
    result = []
    print('Checking naive files:', end='\n    ')
    for em in split_list:
        result.append(p.apply_async(naive_check, args=(em, img_path, of_path, dict_path)))
    p.close()
    p.join()
    dict_ok = True
    for res in result:
        dict_ok = dict_ok and res.get()
    if dict_ok is True:
        print('Naive check (video_frame/of_jpg/of_dict) DONE.')
    else:
        exit('\nZBH: dict check FALSE, see output.')


def of_dict_reader(of_dict_file):
    f = open(of_dict_file, 'r')
    dict_str = f.read()
    of_dict = eval('{' + dict_str + '}')
    f.close()
    return of_dict


def create_tfrecords_list(a_list, save_path, tf_base='data.tfrecords-'):
    tfrecords_list = []
    for file in a_list:  # ['YoYo/v_YoYo_g25_c04.avi', '2'] \ ['YoYo/v_YoYo_g25_c04.avi']
        [_, name] = os.path.splitext(file[0])[0].split('/')  # ['ApplyLipstick','v_ApplyLipstick_g25_c04']
        tf_name = os.path.join(save_path, (tf_base + name))  # save_path\data.tfrecords-v_ApplyLipstick_g25_c04
        tfrecords_list.append(tf_name)
    return tfrecords_list


def make_feature_eval_str(num):
    of_stack_eval_str = ''
    for n in range(num):
        of_ext = ['_x', '_y']
        for xy in range(2):
            of_name = 'of_' + str(n) + of_ext[xy]
            of_eval_str = """'%s': tf.FixedLenFeature([],tf.string),""" % of_name
            of_stack_eval_str = of_stack_eval_str + of_eval_str
    read_tfrecord_base = '''tf.parse_single_example(
                        serialized_example,
                        features={
                            'root':tf.FixedLenFeature([],tf.string),
                            'name':tf.FixedLenFeature([],tf.string),
                            'idex':tf.FixedLenFeature([],tf.string),
                            'label':tf.FixedLenFeature([],tf.int64),
                            'image_raw':tf.FixedLenFeature([],tf.string),
                            'of_min_max_raw': tf.FixedLenFeature([],tf.string),
                        })'''
    read_tfrecord = read_tfrecord_base[:-3] + of_stack_eval_str + '})'
    return read_tfrecord
