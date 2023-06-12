import os
from tqdm import tqdm
import torch.nn.functional as F
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from copy import deepcopy
from torch.utils.tensorboard import SummaryWriter
from warmup_scheduler import GradualWarmupScheduler

from tools.utils import *
from models.B2SModel import B2SModel


from prompts import *

import pdb

class B2STrainer():
    def __init__(self, args):
        self.args = args
        self.model = B2SModel(self.args)
        self.device = torch.device("cuda", self.args.rank)
        self.model.to(self.device)
        
        if self.args.world_size > 1:
            # initialize DDP
            torch.distributed.init_process_group(backend='nccl', init_method='env://', world_size=self.args.world_size, rank=self.args.rank)
            # apply DDP
            # self.model.student = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.model.student)
            self.model = DDP(self.model, device_ids=[self.args.rank], output_device=self.args.rank)
            self.model = self.model.module
            print("Multi-GPUS are used, current device:{}!".format(self.args.rank))

        self.init_paths()

    @torch.no_grad()
    def init_paths(self):
        self.cwd = os.getcwd()
        if self.args.dataset == 'miniImageNet':
            exp_path, dataset_name = 'mini', 'miniimagenet'
            self.class_names, self.templates = None, None
        elif self.args.dataset == 'cifar100':
            exp_path, dataset_name = 'cifar', 'cifar100'
            self.class_names, self.templates = cifar100['classes'], cifar100['templates']
        elif self.args.dataset == 'cub_200':
            exp_path, dataset_name = 'cub', 'cub-200-2011'
            self.class_names, self.templates = cub_200['classes'], cub_200['templates']
        elif self.args.dataset == 'ImageNet':
            exp_path, dataset_name = 'ImageNet', 'ImageNet'
            self.class_names, self.templates = ImageNet['classes'], ImageNet['templates']
        else:
            AssertionError(f"Invalid dataset {self.args.dataset} in Big2SmallTrainer")

        self.dataset_path   = os.path.join(self.cwd, "datasets/{}".format(dataset_name)) # the path of dataset
        self.data_prefix    = exp_path + '_' # prefix of CSV file, e.g. the 'cifar_' in 'cifar_full.csv'

        # paths and writers
        self.work_dir       = os.path.join(self.cwd, "experiments/B2S/{}/{}".format(exp_path, self.args.storage_folder))  # path used to store experimental results
        check_dir(self.work_dir)
        self.model_save_path = os.path.join(self.work_dir, 'b2s.pth')
        self.model_save_key = 'B2S'
        self.writer = SummaryWriter(os.path.join(self.work_dir, 'tensorboard'))
        self.train_log_path = os.path.join(self.work_dir, 'train.log')
        self.test_log_path  = os.path.join(self.work_dir, 'test.log')
    
        # the embedding of all labels
        self.text_embs      = self.model.pre_encode_txt(self.class_names, self.templates)

        self.image_size = 224
        # if 'ViT-L/14' in self.args.tea_arch_name:
        #     self.image_size = 224
        # else:
        #     self.image_size = 224


    #################  get data  #################

    def get_train_loader(self):
        self.args.sampler   = 'std'
        self.args.state     = 'train'
        # self.args.state     = 'test'
        self.args.used_data = self.data_prefix + 'full'
        # self.args.used_data = self.data_prefix + 'base'
        loader = get_dataloader(self.args, image_size=self.image_size)

        return loader

    def get_test_loader(self):
        self.args.sampler   = 'std'
        self.args.state     = 'test'
        self.args.used_data = self.data_prefix + 'test'
        loader = get_dataloader(self.args, image_size=self.image_size)
        
        return loader

    #################  training process  #################
    def optim_pre_op(self):
        # set optim target
        optim_target = [{'params': self.model.parameters(),'lr':self.args.lr}]

        # freeze the teacher model
        self.model.teacher.eval()
        for p in self.model.teacher.parameters():
            p.requires_grad = False
        
        return optim_target
    
    def get_optim(self):
        optim_target = self.optim_pre_op()

        # set optimizer
        if self.args.optimizer == 'sgd':
            optimizer = torch.optim.SGD(optim_target, momentum=self.args.momentum, weight_decay=self.args.wd, nesterov=self.args.nesterov)
        elif self.args.optimizer == 'adam':
            optimizer = torch.optim.Adam(optim_target, weight_decay=self.args.wd)
        else:
            raise ValueError(f"Invalid optimizer:{self.args.optimizer}")
        
        # set scheduler
        if self.args.scheduler == 'SLR':
            scheduler = torch.optim.lr_scheduler.StepLR(
                                    optimizer, step_size=self.args.step_size, gamma=self.args.gamma)
        elif self.args.scheduler == 'MSLR':
            scheduler = torch.optim.lr_scheduler.MultiStepLR(
                                    optimizer, milestones=self.args.milestones, gamma=self.args.gamma)
        elif self.args.scheduler == 'CosineAnnealingLR':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, self.args.epoch)
        else:
            raise ValueError(f"Invalid scheduler {self.args.scheduler}")
        
        # if self.args.world_size > 1:
        #     print("Warm up enabled till epoch 5")
        #     scheduler_wp = GradualWarmupScheduler(optimizer, multiplier=torch.cuda.device_count(), total_epoch=5, after_scheduler=scheduler)
        #     return optimizer, scheduler_wp
        return optimizer, scheduler


    def get_loss(self, data, label, enable_text: bool=True):
        loss_tmp = 0
        # get featurs from the student and teacher models
        all_feats = self.model.get_distill_feats(data, self.text_embs)
        stu_vis   = all_feats['stu_vis_feats']
        tea_vis   = all_feats['tea_vis_feats']

        # calc distillation loss based on visual feature
        loss_vis  = F.mse_loss(stu_vis, tea_vis, reduction='mean')
        loss_tmp  += loss_vis

        # calc contrastive loss
        if enable_text:
            tea_txt = all_feats['tea_txt_feats']
            logits   = F.linear(F.normalize(stu_vis, p=2, dim=-1), F.normalize(tea_txt, p=2, dim=-1))
            loss_con = F.cross_entropy(self.args.temperature*logits, label)
            loss_tmp += loss_con

            # # CLIP contrast
            # logits_   = F.linear(F.normalize(tea_vis, p=2, dim=-1), F.normalize(tea_txt, p=2, dim=-1))
            # loss_con_ = F.cross_entropy(self.args.temperature*logits_, label)
            # loss_tmp += loss_con_
            

        return loss_tmp


    def train_epoch(self, train_loader, epoch:int=0):
        if self.args.rank == 0:
            log(self.train_log_path, 'Train Epoch:{}\tLearning Rate:{:.6f}'.format(epoch, self.scheduler.get_last_lr()[0]))

        self.model.train()
        self.model.teacher.eval()

        train_losses = []
        for i, batch in enumerate(tqdm(train_loader)):
            # get data
            # ImageNet (tensor, [int,int,..])
            data, label = [_.to(self.device) for _ in batch]
            # pdb.set_trace()

            # get loss
            loss = self.get_loss(data, label)

            # optimize the model
            self.optimizer.zero_grad(); loss.backward(); self.optimizer.step()
            train_losses.append(loss.item())

            if self.args.world_size > 1:
                # [cuda0:[1, 2, 3],cuda1:[2,3,4]]
                tmp = torch.FloatTensor(train_losses).to(self.device)
                lists = [torch.zeros_like(tmp).to(self.device) for _ in range(self.args.world_size)]
                dist.all_gather(lists, tmp)
                gathered_data = torch.cat(lists)

                # loss:cuda0 + loss:cuda1 + ...
                dist.all_reduce(loss)
                loss /= self.args.world_size
                loss_avg = torch.mean(gathered_data).item()
            else:
                loss_avg = np.mean(np.array(train_losses))
            
            if self.args.rank==0:
                self.writer.add_scalar('loss', loss_avg, self.writer_axis_x)
                self.writer_axis_x += 1
                if i % 20 == 0:
                    log(self.train_log_path, 'Epoch:{}\tBatch:[{}/{}]\tLoss:{:.4f} % ({:.4f})\t'.format(epoch, i, len(train_loader), loss_avg, loss))
        self.scheduler.step()

    def train(self):
        # get optimizer
        self.optimizer, self.scheduler = self.get_optim()

        # get dataloader
        train_loader = self.get_train_loader()

        # wirte the archetecture of the model, optimizer and scheduler
        if self.args.rank == 0:
            log(self.train_log_path, str(self.model), mode='w+')
            log(self.train_log_path, str(self.optimizer), mode='w+')
            log(self.train_log_path, str(self.scheduler), mode='w+')
            
        # train process
        max_acc = 0
        timer = Timer()
        self.writer_axis_x = 0
        for epoch in range(1, self.args.epoch+1):
            if self.args.world_size > 1:
                train_loader.sampler.set_epoch(epoch-1)
            self.train_epoch(train_loader, epoch)

            if epoch % 5 == 0:
                acc = self.test()
                if max_acc < acc:
                    max_acc = acc
                    if self.args.rank == 0:
                        torch.save({self.model_save_key: self.model.state_dict()}, self.model_save_path)


    #################  inference process  #################
    def get_logits(self, data):
        # get feats 
        stu_vis     = self.model.encode_image_student(data)

        # compute sim
        text_embs = self.model.project_tea_txt(self.text_embs)
        logits    = F.linear(F.normalize(stu_vis, p=2, dim=-1), F.normalize(text_embs, p=2, dim=-1))

        return logits

    @torch.no_grad()
    def test(self, reload:bool=False):
        if reload:
            self.model = load_trained_paras(self.model_save_path, [self.model], [self.model_save_key])[0]
        self.model.eval()

        test_loader = self.get_test_loader()

        test_accs = []
        for i, batch in enumerate(tqdm(test_loader)):
            # get data
            data, label = [_.to(self.device) for _ in batch]

            # get logits
            logits      = self.get_logits(data)

            # compute acc
            preds       = torch.argmax(torch.softmax(logits, dim=-1), dim=-1).reshape(-1)
            acc         = 100 * preds.eq(label).float().mean()
            test_accs.append(acc.item())


        if self.args.world_size > 1:
            # all acc
            tmp = torch.FloatTensor(test_accs).to(self.device)
            lists = [torch.zeros_like(tmp).to(self.device) for _ in range(self.args.world_size)]
            dist.all_gather(lists, tmp)
            test_accs = torch.cat(lists).cpu()
            
        avg = np.mean(np.array(test_accs))

        if self.args.rank == 0:
            log(self.test_log_path, f"acc:{avg}\n")
        
        return avg

    #################  others  #################