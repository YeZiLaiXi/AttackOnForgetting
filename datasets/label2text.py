import csv
import pickle
import os

if __name__ == '__main__':
    file_path = 'cub-200-2011/test.csv'
    l2t_path = 'cub-200-2011/label2text.pkl'
    f_save = open(l2t_path, 'wb')
    if os.path.exists(l2t_path):
        dicts = dict()
        with open(file_path, 'r') as f:
            next(f) # skip first line
            for split in f.readlines():
                split = split.strip('\n').split(',')  # split = [name.jpg label]
                jpg_path = split[0]
                label = split[1]
                text = jpg_path.split('/')[3].split('.')[-1].replace('_', ' ')
                if label not in dicts:
                    dicts[label] = text
        pickle.dump(dicts, f_save)
        f_save.close()
    f_read = open(l2t_path, 'rb')
    saved_dict = pickle.load(f_read)
    f_read.close()
    a = 0
    import pdb
    pdb.set_trace()