import os.path as osp
import numpy as np
import csv
import pdb

if __name__ == '__main__':
    test_path = 'cifar100/cifar_test.csv'
    file_path = 'cifar100/data_info.csv'
    file_path1 = 'cifar100/cifar_full.csv'
    prefix = 'images'

    test_jpgs = []
    with open(test_path) as f:
        next(f)
        for jpg_label in f.readlines():
            jpg, label = jpg_label.strip('\n').split(',')
            jpg_name = jpg.split('/')[-1]
            test_jpgs.append(jpg_name)

    imgs, labels = [], []
    with open(file_path) as f:
        next(f)
        for jpg_label in f.readlines():
            jpg, label = jpg_label.strip('\n').split(',')
            jpg_name = jpg.split('/')[-1]
            if jpg_name not in test_jpgs:
                imgs.append(osp.join(prefix, jpg))
                labels.append(int(label))
    
    # label_key = sorted(np.unique(np.array(labels)))  # input ['a','a','b','b','c','c'] -> ['a','b','c'] output
    # label_map = dict(zip(label_key, range(len(label_key))))  # input['a','b','c'] - > {'a': 0, 'b': 1, 'c': 2}output
    # labels = [label_map[x] for x in labels]  # mapped_labels = [0, 0, 1, 1, 2, 2]

    csv_writer = csv.writer(open(file_path1, 'w', encoding='utf-8', newline=""))
    csv_writer.writerow(['image_name', 'image_label'])
    for i, img in enumerate(imgs):
        csv_writer.writerow([img, labels[i]])
    
            