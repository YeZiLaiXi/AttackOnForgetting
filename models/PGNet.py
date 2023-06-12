from .ResNet import resnet18
from .resnet20_cifar import resnet20

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import math

class PGNet(nn.Module):
    def __init__(self, num_features, num_cls):
        super(PGNet, self).__init__()
        self.num_features = num_features
        hdim=self.num_features
        self.slf_attn = MultiHeadAttention(1, hdim, hdim, hdim, dropout=0.5)

    def update_proto(self, inc_proto, base_proto):
        """
        assume the inc proto is 5X512, the old proto is 60X512, 
        then you can first get 5X60
        Next, you can get update proto 5X512
        In the end, you may adjust this by RPM
        inc_proto: [M, d]
        base_proto: [N, d]
        """
        # relations = F.linear(
        #     F.normalize(inc_proto, p=2, dim=-1), F.normalize(base_proto, p=2, dim=-1)) # [M, N]
        # rpj_proto = torch.matmul(relations, base_proto) # [M, d]

        inc_proto = inc_proto.unsqueeze(0)
        base_proto = base_proto.unsqueeze(0)
        rpj_proto = self.slf_attn(inc_proto, base_proto, base_proto).squeeze(0)
        return rpj_proto

    def forward(self, proto, query):
        logits = F.linear(F.normalize(query, dim=-1, p=2), F.normalize(proto, dim=-1, p=2))
        return logits


    def forward_topk(self, protos, que_feats):
        """
        protos: [n_q, n_w,  d]
        query_features: [n_q, d]

        output: [n_q, n_w]
        """
        n_q, n_w, d = protos.size()
        # attn
        comb = torch.cat((
            protos, que_feats.unsqueeze(1)
        ), dim=1) # [n_q, n_w+1, d]
        comb = self.slf_attn(comb, comb, comb)
        proto_attn, query_attn = comb.split(n_w, 1) # proto:[n_q, n_w, d], query:[n_q, 1, d]
        # compute sim
        logits = []
        logit = F.cosine_similarity(query_attn, proto_attn,dim=-1)
        logits.append(logit)
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
