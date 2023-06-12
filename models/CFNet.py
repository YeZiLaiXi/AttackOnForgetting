from .ResNet import resnet18
from .resnet20_cifar import resnet20

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import math

class CFNet(nn.Module):
    def __init__(self, num_features, feature_imprinting=False, metric_mode='cos', trainable_proto=False):
        super(CFNet, self).__init__()
        self.num_features = num_features
        self.feature_imprinting = feature_imprinting
        hdim=self.num_features
        self.slf_attn = MultiHeadAttention(1, hdim, hdim, hdim, dropout=0.5)
        self.metric_mode = metric_mode

        # make the prototypes trainable
        if trainable_proto:
            self.trainable_proto = torch.randn((100, 512), requires_grad=True)
            self.trainable_proto = nn.Parameter(self.trainable_proto)
        if self.metric_mode == 'RN':
            self.RN = nn.Sequential(
                                nn.Linear(1024, 64, bias=False),
                                nn.LayerNorm(64),
                                nn.ReLU(),
                                nn.Dropout(0.5),
                                nn.Linear(64, 64, bias=False),
                                nn.LayerNorm(64),
                                nn.ReLU(),
                                nn.Dropout(0.5),
                                nn.Linear(64, 1, bias=False),
                                )
                                
    def forward(self, proto, query, return_feature=False, return_proto=False, memory=None, update=True):
        # support: [n_w, d]
        # query: [n_q, d]
        if proto.dim() < 3:
            n_w = proto.shape[0]
            proto = proto.unsqueeze(0).repeat(query.shape[0], 1, 1) # [n_q, n_w, d]
            query = query.unsqueeze(1) # [n_q, 1, d]
        else:
            n_w = proto.shape[1]
            query = query.unsqueeze(1) # [n_q, 1, d]
        if update:
            # solution of ours
            if memory is not None:
                memory = memory.unsqueeze(0).repeat(query.shape[0], 1, 1) # [n_q, 100, d]
                # import pdb
                # pdb.set_trace()
                proto = self.slf_attn(proto, memory, memory)
                query = self.slf_attn(query, memory, memory)

            # solution provided by CEC
            else:
                combined = torch.cat([proto, query], 1) # [n_q, n_w+1, d]
                # 
                combined = self.slf_attn(combined, combined, combined)
                proto, query = combined.split(n_w, 1)

                # 79.77
                # proto = self.slf_attn(proto, combined, combined)
                # query = self.slf_attn(query, combined, combined)
        if return_feature:
            return proto, query

        # MN(cos) PN(euc) RN(relation net)
        if self.metric_mode == 'cos':
            logits = F.cosine_similarity(query-torch.mean(query, dim=-1, keepdim=True), 
                                         proto-torch.mean(proto, dim=-1, keepdim=True), dim=-1) 
        elif self.metric_mode == 'euc':
            logits = -torch.pow((query.expand_as(proto)-proto), 2).sum(dim=-1) / proto.shape[-1]
        elif self.metric_mode == 'RN':
            logits = self.RN(torch.cat((query.expand_as(proto), proto), dim=-1)).squeeze()
        return logits

# PCA

        
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
        output = torch.bmm(attn, v)
        return output, attn, log_attn



class MultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super().__init__()
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

        output = self.dropout(self.fc(output))
        output = self.layer_norm(output + residual)

        return output     
