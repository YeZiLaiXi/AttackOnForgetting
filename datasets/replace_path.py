import os
import csv

def replace_content(old_path, new_path=None):
    if new_path is None:
        new_path = old_path
    img_path = []
    img_label = []
    with open(old_path) as f:
        next(f)
        for c in f.readlines():
            p, l = c.strip('\n').split(',')
            img_path.append(p)
            img_label.append(l)
    if os.path.exists(old_path):
        os.remove(old_path)
    csv_saver = csv.writer(open(new_path, 'w', encoding='utf-8', newline=""))
    csv_saver.writerow(['image_name', 'image_label'])
    for i, p in enumerate(img_path):
        p = p.replace('canghe_20220303', 'canghe_20220308')
        csv_saver.writerow([p, img_label[i]])

if __name__ == '__main__':
    old_path = "/mnt/canghe_20220308/wy/FSIL/datasets/rep_test.csv"

    path_dir = ['/mnt/canghe_20220308/wy/FSIL/datasets/cifar100',
                '/mnt/canghe_20220308/wy/FSIL/datasets/cub-200-2011',
                '/mnt/canghe_20220308/wy/FSIL/datasets/miniimagenet']
    
    for d in path_dir:
        for f in os.listdir(d):
            file_path = os.path.join(d, f)
            if os.path.isfile(file_path):
                file_type = f.split('.')[-1]
                if file_type == 'csv':
                    replace_content(file_path)

    print("hello")