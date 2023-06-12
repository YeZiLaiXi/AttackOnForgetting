import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import torchvision

# import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '7'

# import clip
from .CLIP import clip
from .CLIP.simple_tokenizer import SimpleTokenizer as _Tokenizer
from .ResNet import *
from .resnet20_cifar import *
from .Conv4 import *

from timm.models.layers import  trunc_normal_

# from CLIP import clip
# from CLIP.simple_tokenizer import SimpleTokenizer as _Tokenizer
# from ResNet import *


_tokenizer = _Tokenizer()

import pdb



def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


# from timm.models.layers import PatchEmbed

class B2SModel(nn.Module):

    def __init__(self, args):
        super(B2SModel, self).__init__()
        self.args               = args
        self.dataset            = args.dataset
        self.stu_arch_name      = args.stu_arch_name
        self.tea_arch_name      = args.tea_arch_name
        self.projector          = args.projector
        self.enable_pos_emb     = args.enable_pos_emb
        self.enable_cls_tok     = args.enable_cls_tok
        self.apply_successor    = args.apply_successor
        # initi incremental info
        if self.dataset == 'miniImageNet' or self.dataset == 'cifar100':
            self.num_cls = 100
        elif self.dataset == 'cub_200' or self.dataset == 'ImageNet_R':
            self.num_cls = 200
        elif self.dataset == 'ImageNet':
            self.num_cls = 1000
        else:
            raise Exception("Invalid dataset name {}".format(self.dataset))
        
        # init model
        self.dims = 512
        if self.stu_arch_name == 'ResNet18':
            self.student = resnet18(False)
            self.stu_dim = 512
        elif self.stu_arch_name == 'ResNet20':
            self.student = resnet20()
            self.stu_dim = 64
        elif self.stu_arch_name == 'mobilenet_v2':
            model        = torchvision.models.mobilenet_v2()
            self.student = nn.Sequential(*list(model.children())[:-1]) # [-1, 1280, 7, 7]
            self.stu_dim = 1280
        elif self.stu_arch_name == 'shufflenet_v2_x0_5':
            model = torchvision.models.shufflenet_v2_x0_5()
            self.student = nn.Sequential(*list(model.children())[:-1]) # [-1, 1024, 7, 7]
            self.stu_dim = 1024
        self.head = nn.Linear(self.dims, self.num_cls, bias=False)
        
        self.teacher, self.preprocess = clip.load(self.tea_arch_name, device="cpu")
        if 'ViT-L/14' in self.tea_arch_name:
            self.tea_dim = 768
        else:
            self.tea_dim = 512

        if self.projector == 'Linear':
            self.tea_proj_txt  = nn.Linear(self.tea_dim, self.dims, bias=False) # Linear(512, 512) (1024, 512)
            self.tea_proj_vis  = nn.Linear(self.tea_dim, self.dims, bias=False) # Linear(512, 512) (1024, 512)
            self.stu_proj_vis  = nn.Linear(self.stu_dim, self.dims, bias=False) # Linear(512, 512) 
        elif self.projector == 'Transformer':
            self.tea_proj_txt  = MultiHeadAttention(1, self.dims, self.dims, self.dims, dropout=0.5)
            # self.tea_proj_vis  = MultiHeadAttention(1, self.dims, self.dims, self.dims, dropout=0.5)
            self.tea_proj_vis  = nn.Linear(self.dims, self.dims, bias=False) # Linear(512, 512)
            self.stu_proj_vis  = MultiHeadAttention(1, self.dims, self.dims, self.dims, dropout=0.5)

            if self.enable_cls_tok:
                self.cls_token = torch.randn((1, self.dims), requires_grad=True)
                self.cls_token = nn.Parameter(self.cls_token)
                nn.init.normal_(self.cls_token, std=1e-6)
                    
            if self.enable_pos_emb:
                self.pos_embed = nn.Parameter(torch.randn(1, 50, self.dims) * .02)
                trunc_normal_(self.pos_embed, std=.02)
        else:
            self.tea_proj_txt  = nn.Identity()
            self.tea_proj_vis  = nn.Identity()
            self.stu_proj_vis  = nn.Identity()

        self.evovler =  MultiHeadAttention(1, self.dims, self.dims, self.dims, dropout=0.5)
        

    def project_stu(self, stu_vis):
        b, _, H, W  = stu_vis.size()
        feat_flat   = stu_vis.reshape(-1, self.stu_dim, H*W).permute(0, 2, 1) #[-1, 49, 512]

        if self.projector == 'Transformer':
            if self.enable_cls_tok: # trainable class token
                cls_token     = self.cls_token.unsqueeze(0).repeat(b, 1, 1) # [-1, 1, 512]
            else: # fixed class token
                cls_token     = feat_flat.mean(dim=1, keepdim=True)
            feat_conc   = torch.cat((cls_token, feat_flat), dim=1)

            # add pos emb
            if self.enable_pos_emb:
                feat_conc   = feat_conc + self.pos_embed

            # projecting
            outs        = self.stu_proj_vis(feat_conc, feat_conc, feat_conc)
            cls_emb     = outs[:, 0, :] # [-1, 512]
        else: # Linear or indentity
            cls_token     = feat_flat.mean(dim=1)
            cls_emb       = self.stu_proj_vis(cls_token)

        return cls_emb


    def project_tea_vis(self, tea_vis):
        tea_vis = self.tea_proj_vis(tea_vis) # Linear or indentity
        # if self.projector == 'Transformer':
        #     tea_vis = self.tea_proj_vis(tea_vis.unsqueeze(1), tea_vis.unsqueeze(1), tea_vis.unsqueeze(1)).squeeze()
        # else:
        #     tea_vis = self.tea_proj_vis(tea_vis) # Linear or indentity
        return tea_vis


    def project_tea_txt(self, tea_text):
        if self.projector == 'Transformer':
            tea_text = self.tea_proj_txt(tea_text.unsqueeze(1), tea_text.unsqueeze(1), tea_text.unsqueeze(1)).squeeze()
        else:
            tea_text = self.tea_proj_txt(tea_text) # Linear or indentity
        return tea_text


    def encode_image_student(self, x: torch.Tensor):
        stu_vis = self.student(x) # [-1, 512, 7, 7]
        # pdb.set_trace()
        stu_vis = self.project_stu(stu_vis) # [-1, 512]
        return stu_vis


    def encode_image_teacher(self, x: torch.Tensor):
        with torch.no_grad():
            tea_vis = self.teacher.encode_image(x)
        tea_vis = self.project_tea_vis(tea_vis)
        return tea_vis

    @torch.no_grad()
    def pre_encode_txt(self, classnames, templates):
        tea_txt = []
        for classname in classnames:
            # from clip
            texts        = [template.format(classname) for template in templates] #format with class
            texts        = clip.tokenize(texts).to(self.args.rank) #tokenize
            text_embs    = self.teacher.encode_text(texts) #embed with text encoder
            text_emb     = text_embs.mean(dim=0)
            tea_txt.append(text_emb)
        tea_txt = torch.stack(tea_txt).type(torch.float32)
        return tea_txt

    
    def get_distill_feats(self, imgs, tea_txt=None):
        feats       = {}

        # get featurs from the student model
        feats['stu_vis_feats'] = self.encode_image_student(imgs) # [-1, 512]
        
        # pdb.set_trace()

        #1. get visual feaure from teacher model
        feats['tea_vis_feats'] = self.encode_image_teacher(imgs) # [-1, 512]

        #2. get text feature from teacher model
        if tea_txt != None:
            feats['tea_txt_feats'] = self.project_tea_txt(tea_txt) # [|names|, 512]
        return feats
    
    def evovlution(self, imgs, tea_txt):
        # encode and project stu
        stu_vis    = self.encode_image_student(imgs) # [-1, 512]
        

        # project tx
        proj_txt   = self.project_tea_txt(tea_txt) # [|names|, 512]

        # evolve txt and vis
        expand_vis = stu_vis.unsqueeze(1) # [-1, 1, 512]
        expand_txt = proj_txt.unsqueeze(0).repeat(expand_vis.shape[0], 1, 1) # [-1, |names|, 512]
        conc_feats = torch.cat((expand_txt, expand_vis), dim=1) #[-1, |name|+1, 512]
        updt_feats = self.evovler(conc_feats, conc_feats, conc_feats) #[-1, |name|+1, 512]

        # calc sim
        updt_txt, upd_vis   = updt_feats.split(tea_txt.shape[0], 1)
        logits              = F.cosine_similarity(upd_vis, updt_txt, dim=-1) 

        return logits

    
    def forward(self, imgs, tea_txt=None, linear: bool=False):

        x = self.encode_image_student(imgs)

        # zero-shot 
        if tea_txt is not None:
            tea_text = self.project_tea_txt(tea_txt)
            logits = F.linear(F.normalize(x, p=2, dim=-1), F.normalize(tea_text, p=2, dim=-1)) 

        # other classifiers
        else:
            if linear: # linear classifier
                logits = self.head(x)
            else: # cosine classifer
                logits = F.linear(F.normalize(x, p=2, dim=-1), F.normalize(self.head.weight, p=2, dim=-1)) 
        return logits
 


       
class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v):
        attn = torch.bmm(q, k.transpose(1, 2))
        attn = attn / self.temperature
        log_attn = F.log_softmax(attn, 2)
        attn = self.softmax(attn)
        attn = self.dropout(attn)
        # modified
        # import pdb
        # pdb.set_trace()
        # value, idx = torch.sort(attn, dim=-1)
        # selected_idx = idx[:, :, :5].squeeze()
        # mask = torch.zeros(*idx.size()).to(idx.device)
        # for j in range(selected_idx.shape[0]):
        #     mask[j, :, selected_idx[j]] = 1
        output = torch.bmm(attn, v)
        return output, attn, log_attn



class MultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1, w_res: bool=True):
        super().__init__()
        self.w_res = w_res
        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)
        nn.init.normal_(self.w_qs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_ks.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_vs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_v)))

        self.attention = ScaledDotProductAttention(temperature=np.power(d_k, 0.5))
        self.layer_norm = nn.LayerNorm(d_model)

        self.fc = nn.Linear(n_head * d_v, d_model)
        nn.init.xavier_normal_(self.fc.weight)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v):

        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, _ = q.size()
        sz_b, len_k, _ = k.size()
        sz_b, len_v, _ = v.size()

        residual = q
        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        q = q.permute(2, 0, 1, 3).contiguous().view(-1, len_q, d_k)  # (n*b) x lq x dk
        k = k.permute(2, 0, 1, 3).contiguous().view(-1, len_k, d_k)  # (n*b) x lk x dk
        v = v.permute(2, 0, 1, 3).contiguous().view(-1, len_v, d_v)  # (n*b) x lv x dv
        output, attn, log_attn = self.attention(q, k, v)

        output = output.view(n_head, sz_b, len_q, d_v)
        output = output.permute(1, 2, 0, 3).contiguous().view(sz_b, len_q, -1)  # b x lq x (n*dv)
        if self.w_res:
            output = self.dropout(self.fc(output))
            output = self.layer_norm(output + residual)

        return output 
        # return output, attn # delete later    


# if __name__ == '__main__':
#     model = B2SModel().cuda()
#     images = torch.randn(2, 3, 224, 224).cuda()
#     cls_names = ['cat', 'dog']
#     tmplates = ['a photo of a {}.']
#     stu_vis = model.encode_image_student(images) # [2, 512, 7, 7]
#     tea_vis = model.encode_image_teacher(images) # [2, 512]
#     feats   = model.get_distill_feats(images, cls_names, tmplates)
#     pdb.set_trace()
#     stu_upd = model.project_stu(stu_vis) # [2, 512]
#     tea_text = model.pre_encode_txt(cls_names, tmplates) # [2, 512]

