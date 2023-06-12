import numpy as np
import torch
from torch.utils.data import Sampler
import math
from copy import deepcopy

__all__ = ['Data_Samplers']

# import pdb
# pdb.set_trace()

class Data_Samplers(Sampler):

    def __init__(self, label, cls_list=None, rank_dicts=False,
                 batch_size=None, shuffle=False, drop_last=False, 
                 tasks=None, n_way=None, n_shot=None, n_query=None):
        """
        sampler for standard, incremental, and few-shot

        @para label: the label space of the dataset
        @para cls_list: if not None, then the label of sampled must belong to the cls_list
        @para rank_dicts: if True, then rank the dict {label:index} by ascending order

        # standard sampler setting, also can be used for incremental
        @para batch_size: the num of samples of each batch
        @para shuffle: if True, then shuffle data
        @para drop_last: if True, then drop the end data which can not consistute a batch or a task

        # few-shot sampler setting, also can be used for incremental
        @para tasks: default is None, 
                     if tasks=-1, then the num of task is determined based on the size of dataset
                     if tasks > 0, it's the num of sampled tasks 
        @para n_way: the num of classes of each task
        @para n_shot: the num of supports of each task
        @para n_query: the num of querys of each task
        """
        self.label = label
        self.cls_list = cls_list
        self.rank_dicts = rank_dicts
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.tasks = tasks
        self.n_way = n_way
        self.n_shot = n_shot
        self.n_query = n_query
        self.shots = [self.n_shot, self.n_query]

        if self.tasks is not None:
            if self.n_way is None or self.n_shot is None or self.n_query is None:
                raise ValueError("When sample tasks, n_way and n_shot and n_query should not be NoneType!")

        self.label_inds = {}
        for i, class_id in enumerate(label):
            if class_id not in self.label_inds:
                self.label_inds[class_id] = []
            self.label_inds[class_id].append(i)

        # pop keys not in cls_list
        if self.cls_list is not None:
            for k in list(self.label_inds.keys()):
                if k not in self.cls_list:
                    self.label_inds.pop(k)
        
        # rank the dicts based on the keys by ascending order
        if self.rank_dicts:
            self.label_inds = dict(sorted(self.label_inds.items(), key=lambda d: d[0], reverse=False))

    def __len__(self):
        if self.batch_size is not None:
            pcount = np.array([len(self.label_inds[class_id]) for class_id in self.label_inds])
            if self.drop_last:
                lens = math.floor(sum(pcount) // self.batch_size)
            else:
                lens = math.ceil(sum(pcount) // self.batch_size)

        elif self.tasks is not None:
            if self.tasks > 0: 
                lens = self.tasks
            else: 
                lens = 0
                temp_label_inds = deepcopy(self.label_inds)
                while len(temp_label_inds) >= self.n_way:
                    id_list = []
                    list_class_id = list(temp_label_inds.keys())
                    pcount = np.array([len(temp_label_inds[class_id]) for class_id in list_class_id])
                    batch_class_id = np.random.choice(list_class_id, size=self.n_way, replace=False, p=pcount / sum(pcount))
                    for shot in self.shots:
                        for class_id in batch_class_id:
                            for _ in range(shot):
                                temp_label_inds[class_id].pop()

                    for class_id in batch_class_id:
                        if len(temp_label_inds[class_id]) < sum(self.shots):
                            temp_label_inds.pop(class_id)
                    lens += 1
        else:
            AssertionError("batch_size and tasks can not all be None")

        return lens

    def __iter__(self):
        temp_label_inds = deepcopy(self.label_inds)

        # standard sampler
        if self.batch_size is not None:
            temp_inds = [v for v in temp_label_inds.values()]
            temp_inds = list(np.hstack(temp_inds))

            if self.shuffle:
                np.random.shuffle(temp_inds)
            if self.drop_last:
                batches = math.floor(len(temp_inds) // self.batch_size)
            else:
                batches = math.ceil(len(temp_inds) // self.batch_size)
            for batch_num in range(batches):
                if (batch_num + 1) * self.batch_size <= len(temp_inds):
                    id_list = temp_inds[batch_num * self.batch_size : (batch_num + 1) * self.batch_size]
                else:
                    if not self.drop_last:
                        id_list = temp_inds[batch_num * self.batch_size : len(temp_inds)]
                yield id_list

        # few-shot sampler
        elif self.tasks is not None:
            if self.shuffle:
                for class_id in temp_label_inds:
                    np.random.shuffle(temp_label_inds[class_id])
            # the num of task is determined by users, playback sampling
            if self.tasks > 0: 
                for num in range(self.tasks):
                    id_list = []
                    list_class_id = list(temp_label_inds.keys())
                    batch_class_id = np.random.choice(list_class_id, size=self.n_way, replace=False)
                    for shot in self.shots:
                        for class_id in batch_class_id:
                            sample_id = np.random.randint(low=0, high=len(temp_label_inds[class_id]), size=shot)
                            id_list.append(np.array(temp_label_inds[class_id])[sample_id])
                    id_list = list(np.hstack(id_list))
                    yield id_list
            # the num of task is determined by the size of data, no playback sampling
            else:   
                while len(temp_label_inds) >= self.n_way:
                    id_list = []
                    list_class_id = list(temp_label_inds.keys())
                    pcount = np.array([len(temp_label_inds[class_id]) for class_id in list_class_id])
                    batch_class_id = np.random.choice(list_class_id, size=self.n_way, replace=False, p=pcount / sum(pcount))
                    for shot in self.shots:
                        for class_id in batch_class_id:
                            for _ in range(shot):
                                id_list.append(temp_label_inds[class_id].pop())

                    for class_id in batch_class_id:
                        if len(temp_label_inds[class_id]) < sum(self.shots):
                            temp_label_inds.pop(class_id)

                    yield id_list
        else:
            AssertionError("batch_size and tasks can not all be None")
   
# if __name__ == '__main__':
#     label = [1, 1, 1, 0, 0, 0, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 6, 7, 7, 7, 8, 8, 8, 9, 9, 9, 9, 9]
#     sample = Data_Samplers(label, cls_list=[0, 1, 2], rank_dicts=True, tasks=10, n_way=2, n_shot=2, n_query=1)
#     # sample = Data_Samplers(label, batch_size=1)
#     batches = sample.__iter__()
#     print(sample.__len__())
#     # for b in batches:
#     #     print(b)