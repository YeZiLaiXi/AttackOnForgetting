from collections import OrderedDict
from functools import partial

import numpy as np 
import torch
import torch.nn as nn
from torch.nn.functional import softmax, sigmoid

from timm.models import create_model
import torch.nn.functional as F


class L2PNet(nn.Module):
    def __init__(self,
                 model_name,
                 dataset='cub_200',
                 scale_factor=None,
                 loss=None,
                 metric=None,
                 act_name=None,
                 topk=(1,),
                 freeze_bn=False,
                 pull_constraint=None,
                 pretrained=True,
                 num_classes=100,
                 drop_rate = 0.0,
                 drop_path_rate = 0.0,
                 drop_block_rate=None,
                 prompt_length=5,
                 embedding_key="cls",
                 prompt_init="uniform",
                 prompt_pool=True, 
                 prompt_key=True, 
                 pool_size=10,
                 top_k=5,
                 batchwise_prompt=True,
                 prompt_key_init="uniform",
                 head_type="prompt", 
                 use_prompt_mask=False):
        super().__init__()

        if dataset == 'miniImageNet' or dataset == 'cifar_fs':
            num_classes = 100
        elif dataset == 'cub_200':
            num_classes = 200
        else:
            raise Exception("Invalid dataset name {}".format(dataset))
        # Construct model
        self.ori_vit = create_model(model_name=model_name, pretrained=pretrained, num_classes=num_classes,
                                drop_rate=drop_rate, drop_path_rate=drop_path_rate, drop_block_rate=drop_block_rate)
        self.vit = create_model(model_name=model_name, pretrained=pretrained, num_classes=num_classes,
                                drop_rate=drop_rate, drop_path_rate=drop_path_rate, drop_block_rate=drop_block_rate,
                                prompt_length=prompt_length, embedding_key=embedding_key, prompt_init=prompt_init, 
                                prompt_pool=prompt_pool, prompt_key=prompt_key, pool_size=pool_size, top_k=top_k,
                                batchwise_prompt=batchwise_prompt, prompt_key_init=prompt_key_init, head_type=head_type,
                                use_prompt_mask=use_prompt_mask
                                )
        self.scale_factor = scale_factor
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

    def forward(self, img):
        with torch.no_grad():
            if self.ori_vit is not None:
                output = self.ori_vit(img)
                cls_features = output['pre_logits']
            else:
                cls_features = None
        output = self.vit(img, cls_features=cls_features)
        if self.scale_factor is not None:
            feats = output['pre_logits']
            head_weights = self.vit.head.weight
            logits = self.scale_factor * F.linear(F.normalize(output['pre_logits'], dim=-1, p=2), 
                                                  F.normalize(self.vit.head.weight, dim=-1, p=2))
            output['logits'] = logits
        return output['logits'], output['reduce_sim']