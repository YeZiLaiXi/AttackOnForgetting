import os
import csv
import scipy.io as io
import pdb


if __name__ == '__main__':
    path = 'ILSVRC2012_devkit_t12/data/meta.mat'
    data=io.loadmat(path)
    pdb.set_trace()
    print(data)