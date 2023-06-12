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


class PN_ATTN(nn.Module):
    def __init__(self, num_features, num_cls, dataset='miniImageNet'):
        super(PN_ATTN, self).__init__()
        self.backbone = resnet18()
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(num_features, num_cls, bias=False)
        self.l_w = nn.Parameter(torch.Tensor([1, 1]))
        self.num_features = num_features
        hdim=self.num_features
        self.slf_attn = MultiHeadAttention(1, hdim, hdim, hdim, dropout=0.5)
        # self.register_parameter('l_w',self.l_w)
    
    def encode(self, x, avg=True):
        x = self.backbone(x)
        if avg:
            x = self.avgpool(x).squeeze(-1).squeeze(-1)
        return x

    def compute_similarity(self, proto, query, mode='euc'):
        d = query.shape[-1]
        dist = None
        if mode == 'euc':
            AB = torch.matmul(query, proto.transpose(1, 0))
            AA = (query * query).sum(dim=-1, keepdim=True) 
            BB = (proto * proto).sum(dim=-1, keepdim=True).transpose(1, 0)
            dist = AA.expand_as(AB) - 2 * AB + BB.expand_as(AB)
            dist = - dist
        elif mode == 'cos':
            dist = F.linear(F.normalize(query, p=2, dim=-1), F.normalize(proto, p=2, dim=-1))
            # dist = torch.matmul(query, proto.transpose(1, 0))

        else:
            AssertionError("Invalid mode in compute_similarity")
        
        return dist / d

    def forward(self, support_data, support_labels, query_data, n_way=5, mode='euc', aux_rank_loss=False):
        # encode data
        sup_features = self.encode(
            support_data.reshape([-1] + list(support_data.shape[-3:]))).reshape(support_data.shape[0], -1)
        que_features = self.encode(
            query_data.reshape([-1] + list(query_data.shape[-3:]))).reshape(query_data.shape[0], -1) # [n_q, d]

        # get proto
        one_hot_label = one_hot(support_labels.view(-1), n_way).transpose(0, 1)
        proto = torch.matmul(one_hot_label, sup_features)
        proto = proto.div(one_hot_label.sum(dim=-1, keepdims=True).expand_as(proto)) # [n_w, d]
        # attn
        comb = torch.cat((
            proto.unsqueeze(0).repeat(que_features.shape[0], 1, 1), que_features.unsqueeze(1)
        ), dim=1) # [n_q, n_w+1, d]
        comb = self.slf_attn(comb, comb, comb)
        proto_attn, query_attn = comb.split(proto.shape[0], 1) # proto:[n_q, n_w, d], query:[n_q, 1, d]
        # compute sim
        dists = torch.pow(query_attn.repeat(1, proto.shape[0], 1)-proto_attn, 2).sum(dim=-1)
        logits = - dists / query_attn.shape[-1]
        if aux_rank_loss:
            return logits, self.auxrank(sup_features, n_way)
        else:
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
# if __name__ == '__main__':
#     model = ProtoNet(512, 64).cuda()
#     sup_data = torch.randn((25, 3, 84, 84)).cuda()
#     sup_labe = torch.arange(5).reshape(5, 1).repeat(1, 5).view(-1).cuda()
#     que_data = torch.randn((75, 3, 84, 84)).cuda()
#     print(sup_labe)
#     print(model(sup_data, sup_labe, que_data, n_way=5).shape)
