from .ResNet import resnet18
from .resnet20_cifar import resnet20

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

def one_hot(indices, depth):
    encode_indicies = torch.zeros(indices.size() + torch.Size([depth])).cuda()
    index = indices.view(indices.size() + torch.Size([1]))
    encode_indicies = encode_indicies.scatter_(1, index, 1)

    return encode_indicies

class AM(nn.Module):
    def __init__(self, num_features, num_cls, mode='euc', feature_rec=True):
        super(AM, self).__init__()
        self.mode = mode
        self.feature_rec = feature_rec
        self.num_features = num_features
        hdim=self.num_features

        if self.feature_rec:
            self.correct_layer = nn.Linear(num_features, num_features, bias=False)
        self.slf_attn = MultiHeadAttention(1, hdim, hdim, hdim, dropout=0.5)
        self.fc = nn.Linear(num_features, num_cls, bias=False)


    def forward(self, protos, query_features):
        """
        protos: [n_w,  d]
        query_features: [n_q, d]
        """
        # rec bias
        if self.feature_rec:
            proto_rec = self.correct_layer(protos)
            que_rec = self.correct_layer(query_features)
        else:
            proto_rec = protos
            que_rec = query_features
        # attn
        comb = torch.cat((
            proto_rec.unsqueeze(0).repeat(que_rec.shape[0], 1, 1), que_rec.unsqueeze(1)
        ), dim=1) # [n_q, n_w+1, d]
        comb = self.slf_attn(comb, comb, comb)
        proto_attn, query_attn = comb.split(proto_rec.shape[0], 1) # proto:[n_q, n_w, d], query:[n_q, 1, d]

        # compute sim
        if self.mode == 'euc':
            dists = torch.pow(query_attn.repeat(1, proto_rec.shape[0], 1)-proto_attn, 2).sum(dim=-1)
            logits = - dists / query_attn.shape[-1]
        elif self.mode == 'cos':
            logits=F.cosine_similarity(query_attn, proto_attn,dim=-1)

        return logits


    def forward_topk(self, protos, query_features):
        """
        protos: [n_q, n_w,  d]
        query_features: [n_q, d]

        output: [n_q, n_w]
        """
        n_q, n_w, d = protos.size()
        # rec bias
        proto_rec = self.correct_layer(protos.reshape(-1, d)).reshape(n_q, n_w, d) # [n_q*n_w, d]
        que_rec = self.correct_layer(query_features)
        # attn
        comb = torch.cat((
            proto_rec, que_rec.unsqueeze(1)
        ), dim=1) # [n_q, n_w+1, d]
        comb = self.slf_attn(comb, comb, comb)
        proto_attn, query_attn = comb.split(n_w, 1) # proto:[n_q, n_w, d], query:[n_q, 1, d]

        # compute sim
        if self.mode == 'euc':
            dists = torch.pow(query_attn.repeat(1, n_w, 1)-proto_attn, 2).sum(dim=-1)
            logits = - dists / query_attn.shape[-1]
        elif self.mode == 'cos':
            logits=F.cosine_similarity(query_attn, proto_attn,dim=-1)

        return logits

    def auxrank(self, sup_features, n_way):
        sup = sup_features.reshape(n_way, -1, sup_features.shape[-1])
        way = sup.size(0)
        shot = sup.size(1)
        sup = sup/sup.norm(2).unsqueeze(-1)
        L1 = torch.zeros((way**2-way)//2).long().cuda()
        L2 = torch.zeros((way**2-way)//2).long().cuda()
        counter = 0
        for i in range(way):
            for j in range(i):
                L1[counter] = i
                L2[counter] = j
                counter += 1
        s1 = sup.index_select(0, L1) # (s^2-s)/2, s, d
        s2 = sup.index_select(0, L2) # (s^2-s)/2, s, d

        dists = s1.matmul(s2.permute(0,2,1)) # (s^2-s)/2, s, s
        assert dists.size(-1)==shot
        frobs = dists.pow(2).sum(-1).sum(-1)
        return frobs.sum().mul(.03)
      
        
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
