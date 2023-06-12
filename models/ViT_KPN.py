from collections import OrderedDict
from functools import partial

import numpy as np 
import torch
import torch.nn as nn
from torch.nn.functional import softmax, sigmoid

from timm.models import create_model


class KPNet(nn.Module):
    def __init__(self,
                 model_name,
                 pretrained=True,
                 num_classes=0,
                 drop_rate = 0.0,
                 drop_path_rate = 0.0,
                 drop_block_rate=None,
                 prompt_length=5,
                 prompt_init="uniform",
                 head_type="token",
                 dataset='cub_200'):
        super().__init__()
        # Construct model
        self.ori_vit = create_model(model_name=model_name, pretrained=pretrained, num_classes=num_classes,
                                drop_rate=drop_rate, drop_path_rate=drop_path_rate, drop_block_rate=drop_block_rate)
        self.vit = create_model(model_name=model_name, pretrained=pretrained, num_classes=num_classes,
                                drop_rate=drop_rate, drop_path_rate=drop_path_rate, drop_block_rate=drop_block_rate,
                                prompt_length=prompt_length,  prompt_init=prompt_init,  head_type=head_type, pool_size=1
                                )
    
        # used to store specific knowledge
        self.num_features = 768
        if dataset == 'miniImageNet' or dataset == 'cifar_fs':
            self.num_cls, self.base_cls_num, self.inc_cls_num, self.sessions = 100, 60, 5, 9
        elif dataset == 'cub_200':
            self.num_cls, self.base_cls_num, self.inc_cls_num, self.sessions = 200, 100, 10, 11
        else:
            raise Exception("Invalid dataset name {}".format(dataset))
        self.protos = nn.Linear(self.num_features, self.num_cls, bias=False)
        self.heads_vis = nn.ModuleList()
        self.heads_vis.append(nn.Linear(self.num_features, self.base_cls_num, bias=False))
        for i in range(self.sessions-1):
            self.heads_vis.append(nn.Linear(self.num_features, self.inc_cls_num, bias=False))
        
        # freeze query function
        for p in self.ori_vit.parameters():
            p.requires_grad = False
        part=["blocks", "patch_embed", "cls_token", "norm"]

        # freeze part of vit
        for n, p in self.vit.named_parameters():
            if n.startswith(tuple(part)):
                p.requires_grad = False
    
        # freeze bn
        for module in self.modules():
            if isinstance(module, (nn.BatchNorm2d, nn.BatchNorm3d, nn.LayerNorm)):
                module.train(False)

    def forward(self, 
                img,
                memory=None, 
                KPM=None, 
                cap_layer: int=-1,
                upd_layer: int=-1,
                upd_targt: str='none',
                output_from_original: bool=False):
        if output_from_original:
            output = self.ori_vit(img)
        else:
            if cap_layer != -1:
                mem = self.vit.forward_features(img, cap_layer=cap_layer)
                return mem['x']
            elif upd_layer != -1:
                # output = self.ori_vit(img, memory=memory, KPM=KPM, upd_layer=upd_layer, upd_targt=upd_targt)
                output = self.vit(img, memory=memory, KPM=KPM, upd_layer=upd_layer, upd_targt=upd_targt)
            else:
                output = self.vit(img)
        return output['logits']