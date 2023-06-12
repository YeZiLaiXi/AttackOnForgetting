import os
import shutil
import re
import pickle
import csv
import PIL.Image as Image
import numpy as np
from os import listdir
from os.path import isfile, isdir, join
import random

"""
miniImageNet structure:
miniImageNet
    class_dict
            label:image_index
    image_data
"""


# def MiniImageNet_CSV_Generator():
#     cwd = os.getcwd()
#     dataset_dir = join(cwd, 'mini-imagenet')
#     # print(dataset_dir)
#     split_type = ['train', 'val', 'test']
#     image_save_dir = []

#     for _type in split_type:
#         path = dataset_dir + '/' + _type
#         image_save_dir.append(path)

#     # print(image_save_dir)   # ['./mini-imagenet/train', './mini-imagenet/val', './mini-imagenet/test']
#     for _path in image_save_dir:
#         if os.path.exists(_path):
#             shutil.rmtree(_path)
#             os.mkdir(_path)
#         else:
#             os.mkdir(_path)

#     for file in listdir(dataset_dir):
#         pattern1 = r'[.]'
#         split1 = re.split(pattern1, file)
#         if split1[-1] == 'pkl':
#             ori_data_path = dataset_dir + '/' + file
#             ori_data = pickle.load(open(ori_data_path, 'rb'))
#             labels = ori_data['class_dict']
#             datas = ori_data['image_data']
#             pattern2 = r'[-|.]'
#             split2 = re.split(pattern2, file)
#             csv_saver_path = join(dataset_dir, split2[-2]+'.csv')
#             if os.path.exists(csv_saver_path):
#                 os.remove(csv_saver_path)
#             csv_saver = csv.writer(open(csv_saver_path, 'w', encoding='utf-8', newline=""))
#             csv_saver.writerow(['image_name', 'image_label'])
#             for label in labels:
#                 for image_id in labels[label]:
#                     jpg_name = label + '_' + str(image_id) + '.jpg'
#                     csv_saver.writerow([join(dataset_dir, split2[-2], jpg_name), label])
#                     image = Image.fromarray(datas[image_id])
#                     image.save(join(dataset_dir, split2[-2], jpg_name))

#     return None


def MiniImageNet_CSV_Generator():
    cwd = os.getcwd()
    dataset_dir = join(cwd, 'mini-imagenet')
    for fn in listdir(dataset_dir):
        f_path = os.path.join(dataset_dir, fn) # path + 'train'/'val'/'test'
        if os.path.isfile(f_path):
            continue
        csv_wp = os.path.join(dataset_dir, fn + '.csv') # path + 'train'/'val'/'test' + '.csv'
        if os.path.exists(csv_wp):
            os.remove(csv_wp)
        csv_writer = csv.writer(open(csv_wp, 'w', encoding='utf-8', newline=""))
        csv_writer.writerow(['image_name', 'image_label'])
        for imgs in listdir(f_path):
            img_path = join(f_path, imgs)
            label = imgs.split('_')[0]
            csv_writer.writerow([img_path, label])
    return None

def tieredImageNet_CSV_Generator():
    cwd = os.getcwd()
    dataset_dir = join(cwd, 'tiered_imagenet/tiered_imagenet')
    for fn in listdir(dataset_dir):
        f_path = os.path.join(dataset_dir, fn) # path + 'train'/'val'/'test'
        if os.path.isfile(f_path):
            continue
        csv_wp = os.path.join(dataset_dir, fn + '.csv') # path + 'train'/'val'/'test' + '.csv'
        if os.path.exists(csv_wp):
            os.remove(csv_wp)
        csv_writer = csv.writer(open(csv_wp, 'w', encoding='utf-8', newline=""))
        csv_writer.writerow(['image_name', 'image_label'])
        for labels in listdir(f_path):
            s_path = join(f_path, labels)
            if os.path.isfile(s_path):
                continue
            for img in listdir(s_path):
                postfix = img.split('.')[-1]
                if postfix == 'jpg' and img[0] != '.':
                    csv_writer.writerow([join(s_path, img), labels])
    return None


def Cifar_FS_CSV_Generator(): 
    cwd = os.getcwd()
    dataset_dir = join(cwd, 'cifar-fs/CIFAR-FS/cifar100')
    split_txt_dir = os.path.join(dataset_dir, 'splits', 'bertinetto')
    for txt in os.listdir(split_txt_dir):
        txt_path = os.path.join(split_txt_dir, txt)
        pattern = r'[.]'
        split1 = re.split(pattern, txt)
        if split1[0] == 'train':
            csv_saver_path = dataset_dir + '/' + 'train.csv'
        if split1[0] == 'val':
            csv_saver_path = dataset_dir + '/' + 'val.csv'
        if split1[0] == 'test':
            csv_saver_path = dataset_dir + '/' + 'test.csv'
        
        if os.path.exists(csv_saver_path):
            os.remove(csv_saver_path)

        csv_saver = csv.writer(open(csv_saver_path, 'w', encoding='utf-8', newline=""))
        csv_saver.writerow(['image_name', 'image_label'])

        with open(txt_path, 'r') as f:
            for text in f.readlines():
                text = text.strip('\n')
                folder_path = os.path.join(dataset_dir, 'data', text)
                for image in os.listdir(folder_path):
                    img_path = join(folder_path, image)
                    if isfile(img_path) and image[0] != '.':
                        csv_saver.writerow([img_path, text])
    return None


def Cub_200_2011_CSV_Generator():
    cwd = os.getcwd()
    data_path = join(cwd, 'cub-200-2011/Caltech-UCSD Birds-200-2011/CUB_200_2011/images')
    savedir = './'
    dataset_list = ['train','val','test']
    folder_list = [f for f in listdir(data_path) if isdir(join(data_path, f))]
    folder_list.sort()
    label_dict = dict(zip(folder_list,range(0, len(folder_list))))

    classfile_list_all = [] 

    for i, folder in enumerate(folder_list):
        folder_path = join(data_path, folder)
        classfile_list_all.append( [ join(folder_path, cf) for cf in listdir(folder_path) if (isfile(join(folder_path,cf)) and cf[0] != '.')])
        random.shuffle(classfile_list_all[i])

    for dataset in dataset_list:
        file_list = []
        label_list = []
        for i, classfile_list in enumerate(classfile_list_all):
            if 'train' in dataset:
                if (i%2 == 0):
                    file_list = file_list + classfile_list
                    label_list = label_list + np.repeat(i, len(classfile_list)).tolist()
            if 'val' in dataset:
                if (i%4 == 1):
                    file_list = file_list + classfile_list
                    label_list = label_list + np.repeat(i, len(classfile_list)).tolist()
            if 'test' in dataset:
                if (i%4 == 3):
                    file_list = file_list + classfile_list
                    label_list = label_list + np.repeat(i, len(classfile_list)).tolist()
        csv_saver_path = join(join(cwd, 'cub-200-2011'), dataset + '.csv')
        if os.path.exists(csv_saver_path):
            os.remove(csv_saver_path)
        csv_saver = csv.writer(open(csv_saver_path, 'w', encoding='utf-8', newline=""))
        csv_saver.writerow(['image_name', 'image_label'])
        for ind in range(len(label_list)):
            csv_saver.writerow([file_list[ind], label_list[ind]])

    return None

# import pdb
# pdb.set_trace()

if __name__ == '__main__':
    # MiniImageNet_CSV_Generator()

    # tieredImageNet_CSV_Generator()

    Cub_200_2011_CSV_Generator()

    # Cifar_FS_CSV_Generator()

    # data = []
    # ori_labels = []
    # './cub-200-2011/test.csv'
    # './cub-200-2011/train.csv'
    # './cub-200-2011/val.csv'
    # with open('./cub-200-2011/train.csv', 'r') as f:
    #     next(f)  # skip first line
    #     for split in f.readlines():
    #         split = split.strip('\n').split(',')  # split = [name.jpg label]
    #         data.append(split[0])  # name.jpg
    #         ori_labels.append(split[1])  # label
    # uni = np.unique(np.array(ori_labels))
    # print("")
    
#     mini_dir = './mini-imagenet'
#     cifar_dir = './cifar-fs/CIFAR-FS/cifar100'
#     cub_dir = './cub-200-2011/Caltech-UCSD Birds-200-2011/CUB_200_2011'
#     # MiniImageNet_CSV_Generator(mini_dir)
#     # Cifar_FS_CSV_Generator(cifar_dir)
#     # Cub_200_2011_CSV_Generator(cub_dir)
#     print('hello')


    # from PIL import Image
    # img = Image.open('/wangye/datasets/mini-imagenet/test/n07613480_11999.jpg')
    # print(np.array(img).shape)
    # print(len(os.listdir('/wangye/datasets/mini-imagenet/val')))
