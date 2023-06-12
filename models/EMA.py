# EMA 类 通过记录旧模型的参数，在新模型更新的时候进行步长更新
import copy

import torch
import torch.nn as nn


class EMA():
    def __init__(self, model, decay):
        self.model = model
        self.decay = decay
        self.shadow = {} # old model
        self.backup = {} # new model

    def search_child_module(self, modules, parent_name = ""):
        para_dict = {}
        for k, v in modules.items():
            if isinstance(v, nn.BatchNorm2d):
                para_dict[".".join([parent_name, k, "running_mean"])] = v.running_mean
                para_dict[".".join([parent_name, k, "running_var"])] = v.running_var
                continue
            if isinstance(v, nn.BatchNorm1d):
                para_dict[".".join([parent_name, k, "running_mean"])] = v.running_mean
                para_dict[".".join([parent_name, k, "running_var"])] = v.running_var
                continue
            para_dict.update(self.search_child_module(v._modules, parent_name = ".".join([parent_name, k])))
        return para_dict


    def register(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()
        now_parameters = self.search_child_module(self.model._modules, parent_name = "mm")
        for name, param in now_parameters.items():
            self.shadow[name] = param.data.clone()
        self.backup = copy.deepcopy(self.shadow)


    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()
        now_parameters = self.search_child_module(self.model._modules, parent_name="mm")
        for name, param in now_parameters.items():
            assert name in self.shadow
            new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
            self.shadow[name] = new_average.clone()


    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name].clone()
        now_parameters = self.search_child_module(self.model._modules, parent_name="mm")
        for name, param in now_parameters.items():
            assert name in self.shadow
            self.backup[name] = param.data.clone()
            param.data = self.shadow[name].clone()

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.backup
                param.data = self.backup[name].clone()
        now_parameters = self.search_child_module(self.model._modules, parent_name="mm")
        for name, param in now_parameters.items():
            assert name in self.backup
            param.data = self.backup[name].clone()