from .ResNet import resnet18
from .resnet20_cifar import resnet20

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import math

class PINet(nn.Module):
    def __init__(self, num_features):
        super(PINet, self).__init__()
        self.num_features = num_features
        hdim=self.num_features

        self.slf_attn1 = MultiHeadAttention(1, hdim, hdim, hdim, dropout=0.5)
        self.slf_attn2 = MultiHeadAttention(1, hdim, hdim, hdim, dropout=0.5)
        self.linear = nn.Linear(self.num_features, self.num_features, bias=False)
    
    def get_proto(self, support):
        """
        support: [n_w, shot, d]
        """
        n_w, shot, d = support.size()
        feat_attn = self.slf_attn1(support, support, support) # [n_w, shot, d]
        w = self.linear(feat_attn.reshape(-1, d)).reshape(n_w, shot, d) # [n_w, shot, d]
        proto = w.mean(dim=1) # [n_w, d]

        return proto

    def forward(self, support, query, mode='test'):
        """
        support: [n_w, shot, d] when train or [n_q, n_w, d] when test
        query: [n_q, d]
        """
        if mode == 'train':
            proto = self.get_proto(support) # [n_w, d]
            n_w = proto.shape[0]
            proto = proto.unsqueeze(0).repeat(query.shape[0], 1, 1) # [n_q, n_w, d]
            query = query.unsqueeze(1) # [n_q, 1, d]
        else:
            proto = support
            n_w = proto.shape[1]
            query = query.unsqueeze(1) # [n_q, 1, d]

        # compute_sim
        combined = torch.cat([proto, query], 1) # [n_q, n_w+1, d]
        combined = self.slf_attn2(combined, combined, combined)
        proto_, query = combined.split(n_w, 1)

        # compute sim
        logits = F.cosine_similarity(query, proto_, dim=-1) # 
        if mode == 'train':
            return logits, proto
        else:
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
