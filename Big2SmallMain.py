import os
from tqdm import tqdm
import argparse

# os.environ['CUDA_VISIBLE_DEVICES'] = '0,4,5,6'
# os.environ['CUDA_VISIBLE_DEVICES'] = '7'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

from tools.utils import *


from Big2SmallTrainer import B2STrainer
from SevenTrainer import SevenTrainer
import torch.multiprocessing as mp

import pdb

os.environ['NCCL_SOCKET_IFNAME'] = 'eth0'
os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '15778'


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Unsupported value encountered.')


def set_parameters(parser):
    parser.add_argument('--storage_folder', type=str, default='tmpfile') # tmpfile
    parser.add_argument('--trainer', type=str, default='B2STrainer') # tmpfile
    parser.add_argument('--state', type=str, default='train', choices=['train', 'test'],
                        help='training or testing')
    # main setting
    parser.add_argument('--stu_arch_name', type=str, default='ResNet18', help='The architecture of student model')
    parser.add_argument('--tea_arch_name', type=str, default='ViT-B/32', help='The architecture of teacher model')
    parser.add_argument('--projector', type=str, default="Linear",
                        choices=['Linear', 'Transformer', 'Identity'])
    parser.add_argument('--enable_pos_emb', type=str2bool, default=True)
    parser.add_argument('--enable_cls_tok', type=str2bool, default=True)
    parser.add_argument('--apply_successor', type=str2bool, default=True)

    parser.add_argument('--dataset', type=str, default='cifar100',
                        choices=['miniImageNet', 'tieredImagenet', 'cifar100', 'cub_200', 'ImageNet'],
                        help='datasets')
    parser.add_argument('--used_data', type=str, default='train', choices=['train_val', 'train_val', 'test', 'min_2_cub'],
                        help='only use training set or use both training set and validation set to train model')
    parser.add_argument('--sampler', type=str, default='std', choices=['fsl', 'std', 'task', 'FRN'],
                        help='data sampler')
    parser.add_argument('--workers', type=int, default=4,#4
                        help='num of thread to process image')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--seed', type=int, default=5) # 
    parser.add_argument('--imagenet_train_path', type=str, default="./datasets/ImageNet/ILSVRC2012_img_train")
    parser.add_argument('--imagenet_val_path', type=str, default="./datasets/ImageNet/ILSVRC2012_img_val")

    # ddp
    parser.add_argument('-world_size', type=int, default=1)
    parser.add_argument('-rank', type=int, default=0)

    #  task for training
    parser.add_argument('--tasks', type=int, default=200,  # 2000
                        help='number of tasks used to train in each epoch')
    parser.add_argument('--n-way', type=int, default=5,
                        help='n_way')
    parser.add_argument('--train-shot', type=int, default=5,
                        help='n_shot')
    parser.add_argument('--train-query', type=int, default=15,
                        help='n_query')

    # training: global setting
    parser.add_argument('--epoch', type=int, default=100) # 50
    parser.add_argument('--optimizer', type=str, default='adam')
    parser.add_argument('--lr', type=float, default=0.1,
                        help='learning rate')
    parser.add_argument('--wd', type=float, default=1e-4,
                        help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='weight decay')
    parser.add_argument('--nesterov', type=str2bool, default=True,
                        help='nesterov')
    parser.add_argument('--scheduler', type=str, default='CosineAnnealingLR',
                        help='scheduler')
    parser.add_argument('--milestones', type=int, nargs='+', default=[30, 40, 60, 80]) # [30, 40]
    parser.add_argument('--gamma', type=float, default=0.1)
    parser.add_argument('--step_size', type=int, default=20) # 10


    # pretrain setting
    parser.add_argument('--pretrain', type=str2bool, default=False,
                        help='whether use pre-trained model')
    parser.add_argument('--temperature', type=int, default=16) # 16
    parser.add_argument('--batch_size', type=int, default=128) #128

    return parser

def main_process(rank=0, args=None):
    # set_seed(args.seed+rank)
    set_seed(args.seed)
    args.rank = rank
    if args.trainer == 'B2STrainer':
        trainer = B2STrainer(args)
    elif args.trainer == 'SevenTrainer':
        trainer = SevenTrainer(args)

    if args.pretrain:
        trainer.pretrain()
    elif args.state == 'train':
        trainer.train()
    trainer.test(reload=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='mynet')
    parser = set_parameters(parser)
    args = parser.parse_args()
    print(args)
    args.world_size = torch.cuda.device_count()
    
    if args.world_size > 1:
        mp.spawn(main_process, nprocs=args.world_size, args=(args,))
    else:
        main_process(args=args)
