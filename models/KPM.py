from .ResNet import resnet18
from .resnet20_cifar import resnet20

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import math

class KPM(nn.Module):
    def __init__(self, 
                 num_features, 
                 enable_general_prompt:bool=False, 
                 gen_prompt: bool=False,
                 enable_fusion: bool=False):
        super(KPM, self).__init__()
        self.num_features = num_features
        self.enable_general_prompt  = enable_general_prompt
        self.gen_prompt             = gen_prompt

        hdim    = self.num_features
        w_res   = True
        if self.gen_prompt:
            w_res = False
        
        self.linear1 = nn.Linear(hdim, 768, bias=False)
        # self.linear2 = nn.Linear(512, 512, bias=False)

        if self.gen_prompt: 
            self.dc_l1      = nn.Linear(512, 768, bias=False)
            self.dc_attn    = MultiHeadAttention(1, 768, 768, 768, dropout=0.5, w_res=w_res)
            self.dc_prompt = torch.randn((4, 768), requires_grad=True)
            self.dc_prompt = nn.Parameter(self.dc_prompt)
        else:
            self.slf_attn   = MultiHeadAttention(1, 768, 768, 768, dropout=0.5)
            self.slf_attn2  = MultiHeadAttention(1, 512, 512, 512, dropout=0.5)
            # self.ktm        = nn.Linear(768, 768, bias=False)
        
        if enable_fusion:
            self.fusion_block = MultiHeadAttention(1, hdim, hdim, hdim, dropout=0.5)
    
    
    def fusion(self, vis_parts, sem_parts):
        sem_parts = sem_parts.unsqueeze(1)
        if vis_parts.dim() < 3:
            vis_parts = vis_parts.unsqueeze(1)
        combs       = torch.cat((vis_parts, sem_parts), dim=1) # [n_w, -1, d]
        combs       = self.fusion_block(combs, combs, combs)
        split_node  = combs.shape[1]-1
        sem_proto   = combs.split(split_node, 1)[1].mean(dim=1)
        return sem_proto
        # comb_proto = torch.cat((vis_parts, sem_parts), dim=0).unsqueeze(1) # [n_w, 2, d]
        # comb_proto = self.fusion_block(comb_proto, comb_proto, comb_proto).squeeze(0)
        # proto = comb_proto.split(n_w, 0)[0]
        # return proto


    def prompt_generation(self, x):
        # x: [n_w, d]
        n_w, d = x.size()
        if d == 512:
            x = self.dc_l1(x)
        comb = torch.cat((x, self.dc_prompt), dim=0)
        comb = self.dc_attn(comb.unsqueeze(0), comb.unsqueeze(0),comb.unsqueeze(0)).squeeze(0)
        new_prompts = comb[n_w:]
        return new_prompts

    def forward(self, proto, query, memory=None):
        # support: [n_w, d]
        # query: [n_q, d]
        if proto.dim() < 3:
            n_w     = proto.shape[0]
            proto   = proto.unsqueeze(0).repeat(query.shape[0], 1, 1) # [n_q, n_w, d]  
        else:
            n_w     = proto.shape[1]
            query   = query.unsqueeze(1) # [n_q, 1, d]
        
        if query.dim() < 3:
            query   = query.unsqueeze(1) # [n_q, 1, d]
    
        # solution of ours
        if memory is not None:
            memory  = memory.unsqueeze(0).repeat(query.shape[0], 1, 1) # [n_q, 100, d]
            proto   = self.slf_attn2(proto, memory, memory)
            query   = self.slf_attn2(query, memory, memory)

        # solution provided by CEC
        else:
            combined        = torch.cat([proto, query], 1) # [n_q, n_w+1, d]
            combined        = self.slf_attn2(combined, combined, combined)
            proto, query    = combined.split(n_w, 1)

        logits      = F.cosine_similarity(query, proto, dim=-1) 
        return logits


    def update(self, 
                x=None, 
                memory=None, 
                prefix:bool=False):
        """
        Input:
            x: [-1, d]
            memory: [-1, d]
        Output:
            x: [-1, d]
        """
        # whether update proto based on memory
        if x is not None and memory is not None:
            if x.dim() < 3:
                x = x.unsqueeze(1) # [-1, 1, d]
            if memory.dim() < 3:
                memory_ = memory.unsqueeze(0).repeat(x.shape[0], 1, 1)
            
            if prefix:
                x = self.slf_attn(x, torch.cat((memory_, x), dim=1), torch.cat((memory_, x), dim=1))
            else:
                x  = self.slf_attn(x, memory_, memory_)
            x  = x.squeeze(1)
        else:
            x  = x

        return x
        
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
