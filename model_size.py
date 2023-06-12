

from models.CLIP import clip
from models.CLIP.simple_tokenizer import SimpleTokenizer as _Tokenizer
from models.ResNet import *

from models.Conv4 import Conv4
import torchvision
import torch.nn as nn

from models.resnet20_cifar import resnet20
import numpy as np

def get_model_size(model):
    # 定义总参数量、可训练参数量及非可训练参数量变量
    Total_params = 0
    Trainable_params = 0
    NonTrainable_params = 0

    # 遍历model.parameters()返回的全局参数列表
    for param in model.parameters():
        mulValue = np.prod(param.size())  # 使用numpy prod接口计算参数数组所有元素之积
        Total_params += mulValue  # 总参数量
        if param.requires_grad:
            Trainable_params += mulValue  # 可训练参数量
        else:
            NonTrainable_params += mulValue  # 非可训练参数量

    print(f'Total params: {Total_params/10**6}M')
    print(f'Trainable params: {Trainable_params/10**6}M')
    print(f'Non-trainable params: {NonTrainable_params/10**6}M')

if __name__ == '__main__':
    # model, _ = clip.load('ViT-L/14', device="cpu")
    model = torchvision.models.vgg11_bn()
    print(model)
    # mode_del = nn.Sequential(*list(model.children())[:-1])
    # print(mode_del)
    get_model_size(model)