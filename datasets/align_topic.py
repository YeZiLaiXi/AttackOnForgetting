import os
import csv
import re

import pdb 
pdb.set_trace()

def align_topic_cub():
    cwd = os.getcwd() # /mnt/canghe_20220303/wy/FSIL/datasets
    txt_root = os.path.join(cwd, 'data/index_list/cub200')
    sessions = 11
    base_part, inc_part, test_part = {}, {}, {}
    for sess in range(1, sessions+1):
        cur_train_txt = os.path.join(txt_root, 'session_{}.txt'.format(sess))
        with open(cur_train_txt) as f:
            contens = f.readlines()
            for c in contens:
                label = c.split('/')[-2]
                jpg_name = c.replace('\n', '')
                if sess == 1:
                    if label not in base_part:
                        base_part[label] = []
                    base_part[label].append(jpg_name)
                else:
                    if label not in inc_part:
                        inc_part[label] = []
                    inc_part[label].append(jpg_name)
    cur_test_txt = os.path.join(txt_root, 'test.txt')
    with open(cur_test_txt) as f:
        contens = f.readlines()
        for c in contens:
            label = c.split('/')[-2]
            jpg_name = c.replace('\n', '')
            if label not in test_part:
                test_part[label] = []
            test_part[label].append(jpg_name)
    label_int_map = {}
    for ind, l in enumerate(base_part.keys()):
        label_int_map[l] = ind
    for ind, l in enumerate(inc_part.keys()):
        label_int_map[l] = ind + 100
    
    # initialize csv
    data_path = os.path.join(cwd, 'cub-200-2011')

    # csv_saver_path_base_train = os.path.join(data_path, 'cub_base.csv')
    # csv_saver_path_incr_train = os.path.join(data_path, 'cub_inc.csv')
    # csv_saver_path_test = os.path.join(data_path, 'cub_test.csv')
    
    # for file_path in [csv_saver_path_base_train, csv_saver_path_incr_train, csv_saver_path_test]:
    #     if os.path.exists(file_path):
    #         os.remove(file_path)
    # csv_saver_base = csv.writer(open(csv_saver_path_base_train, 'w', encoding='utf-8', newline=""))
    # csv_saver_base.writerow(['image_name', 'image_label'])
    # csv_saver_incr = csv.writer(open(csv_saver_path_incr_train, 'w', encoding='utf-8', newline=""))
    # csv_saver_incr.writerow(['image_name', 'image_label'])
    # csv_saver_test = csv.writer(open(csv_saver_path_test, 'w', encoding='utf-8', newline=""))
    # csv_saver_test.writerow(['image_name', 'image_label'])

    # for label in base_part.keys():
    #     for jpg_name in base_part[label]:
    #         my_path =  os.path.join(cwd, 'cub-200-2011/Caltech-UCSD Birds-200-2011/CUB_200_2011')
    #         jpg_name = jpg_name.replace('CUB_200_2011', my_path)
    #         csv_saver_base.writerow([jpg_name, label_int_map[label]])
    # for label in inc_part.keys():
    #     for jpg_name in inc_part[label]:
    #         my_path =  os.path.join(cwd, 'cub-200-2011/Caltech-UCSD Birds-200-2011/CUB_200_2011')
    #         jpg_name = jpg_name.replace('CUB_200_2011', my_path)
    #         csv_saver_incr.writerow([jpg_name, label_int_map[label]])
    # for label in test_part.keys():
    #     for jpg_name in test_part[label]:
    #         my_path =  os.path.join(cwd, 'cub-200-2011/Caltech-UCSD Birds-200-2011/CUB_200_2011')
    #         jpg_name = jpg_name.replace('CUB_200_2011', my_path)
    #         csv_saver_test.writerow([jpg_name, label_int_map[label]])

    
    # get all training and testing samples
    train_file = os.path.join(data_path, 'train.csv')
    test_file = os.path.join(data_path, 'test.csv')
    train_part = {}
    cur_train_txt = os.path.join(txt_root, 'train.txt')
    with open(cur_train_txt) as f:
        contens = f.readlines()
        for c in contens:
            label = c.split('/')[-2]
            jpg_name = c.replace('\n', '')
            if label not in train_part:
                train_part[label] = []
            train_part[label].append(jpg_name)
    
    saver_train = csv.writer(open(train_file, 'w', encoding='utf-8', newline=""))
    saver_test = csv.writer(open(test_file, 'w', encoding='utf-8', newline=""))
    saver_train.writerow(['image_name', 'image_label'])
    saver_test.writerow(['image_name', 'image_label'])

    for label in test_part.keys():
        for jpg_name in test_part[label]:
            my_path =  os.path.join(cwd, 'cub-200-2011/Caltech-UCSD Birds-200-2011/CUB_200_2011')
            jpg_name = jpg_name.replace('CUB_200_2011', my_path)
            saver_test.writerow([jpg_name, label_int_map[label]])

    for label in train_part.keys():
        for jpg_name in train_part[label]:
            my_path =  os.path.join(cwd, 'cub-200-2011/Caltech-UCSD Birds-200-2011/CUB_200_2011')
            jpg_name = jpg_name.replace('CUB_200_2011', my_path)
            saver_train.writerow([jpg_name, label_int_map[label]])


def align_topic_mini():
    cwd = os.getcwd() # /mnt/canghe_20220303/wy/FSIL/datasets
    txt_root = os.path.join(cwd, 'data/index_list/mini_imagenet')
    sessions = 9
    base_part, inc_part, test_part = {}, {}, {}
    for sess in range(1, sessions+1):
        cur_train_txt = os.path.join(txt_root, 'session_{}.txt'.format(sess))
        with open(cur_train_txt) as f:
            contens = f.readlines()
            for c in contens:
                label = c.split('/')[-2]
                jpg_name = c.split('/')[-1]
                jpg_name = jpg_name.replace('\n', '')
                if sess == 1:
                    if label not in base_part:
                        base_part[label] = []
                    base_part[label].append(jpg_name)
                else:
                    if label not in inc_part:
                        inc_part[label] = []
                    inc_part[label].append(jpg_name)
    cur_test_txt = os.path.join(txt_root, 'test.txt')
    with open(cur_test_txt) as f:
        contens = f.readlines()
        for c in contens:
            label = c.split('/')[-2]
            jpg_name = c.split('/')[-1]
            jpg_name = jpg_name.replace('\n', '')
            if label not in test_part:
                test_part[label] = []
            test_part[label].append(jpg_name)
    label_int_map = {}
    for ind, l in enumerate(base_part.keys()):
        label_int_map[l] = ind
    for ind, l in enumerate(inc_part.keys()):
        label_int_map[l] = ind + 60
    
    # initialize csv
    data_path = os.path.join(cwd, 'miniimagenet')

    # csv_saver_path_base_train = os.path.join(data_path, 'mini_base.csv')
    # csv_saver_path_incr_train = os.path.join(data_path, 'mini_inc.csv')
    # csv_saver_path_test = os.path.join(data_path, 'mini_test.csv')
    
    # for file_path in [csv_saver_path_base_train, csv_saver_path_incr_train, csv_saver_path_test]:
    #     if os.path.exists(file_path):
    #         os.remove(file_path)
    # csv_saver_base = csv.writer(open(csv_saver_path_base_train, 'w', encoding='utf-8', newline=""))
    # csv_saver_base.writerow(['image_name', 'image_label'])
    # csv_saver_incr = csv.writer(open(csv_saver_path_incr_train, 'w', encoding='utf-8', newline=""))
    # csv_saver_incr.writerow(['image_name', 'image_label'])
    # csv_saver_test = csv.writer(open(csv_saver_path_test, 'w', encoding='utf-8', newline=""))
    # csv_saver_test.writerow(['image_name', 'image_label'])

    # for label in base_part.keys():
    #     for jpg_name in base_part[label]:
    #         abs_path = os.path.join(os.path.join(cwd, 'miniimagenet/images'),jpg_name)
    #         csv_saver_base.writerow([abs_path, label_int_map[label]])
    # for label in inc_part.keys():
    #     for jpg_name in inc_part[label]:
    #         abs_path = os.path.join(os.path.join(cwd, 'miniimagenet/images'),jpg_name)
    #         csv_saver_incr.writerow([abs_path, label_int_map[label]])
    # for label in test_part.keys():
    #     for jpg_name in test_part[label]:
    #         abs_path = os.path.join(os.path.join(cwd, 'miniimagenet/images'),jpg_name)
    #         csv_saver_test.writerow([abs_path, label_int_map[label]])

    
    # get all training and testing samples
    train_file = os.path.join(data_path, 'train.csv')
    test_file = os.path.join(data_path, 'test.csv')
    samples = {}
    for files in [train_file, test_file]:
        with open(files, 'r') as f:
            next(f)
            for c in f.readlines():
                jpg_name, label = c.strip('\n').split(',')
                if label not in samples:
                    samples[label] = []
                samples[label].append(jpg_name)
        os.remove(files)
    
    saver_train = csv.writer(open(train_file, 'w', encoding='utf-8', newline=""))
    saver_test = csv.writer(open(test_file, 'w', encoding='utf-8', newline=""))
    saver_train.writerow(['image_name', 'image_label'])
    saver_test.writerow(['image_name', 'image_label'])

    for l in samples.keys():
        for ind, jpg_name in enumerate(samples[l]):
            jpg_path = os.path.join(os.path.join(cwd, 'miniimagenet/images'), jpg_name)
            if ind < 500:
                saver_train.writerow([jpg_path, label_int_map[l]])
            else:
                saver_test.writerow([jpg_path, label_int_map[l]])

         
if __name__ == '__main__':
    csv_path = '/mnt/canghe_20220308/wy/FSIL/datasets/cifar100/data_info.csv'
    infos = {}
    with open(csv_path) as f:
        next(f)
        for jpg_label in f.readlines():
            jpg, label = jpg_label.strip('\n').split(',')
            if label not in infos:
                infos[label] = []
            infos[label].append(jpg)
        a = 1




        
        

    