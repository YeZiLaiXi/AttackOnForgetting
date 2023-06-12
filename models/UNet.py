from .UNet_parts import *
from .ResNet import resnet18

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import math


class Deconv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Deconv, self).__init__()
        self.de = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=1, stride=2)
        self.conv1 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x_forward, x_backward):
        identi = x_forward
        x = self.de(x_backward)

        diffY = x_forward.size()[2] - x.size()[2]
        diffX = x_forward.size()[3] - x.size()[3]

        x = F.pad(x, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = x + identi
        x = self.relu2(x)
        return x

class final_conv(nn.Module):
    def __init__(self):
        super(final_conv, self).__init__()
        self.ups = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.de = nn.ConvTranspose2d(64, 3, kernel_size=7, stride=2, padding=3)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(3, 3, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(3, 3, kernel_size=3, padding=1)
    

    def forward(self, x):

        x = self.ups(x)
        x = self.de(x)

        diffY = 224- x.size()[2]
        diffX = 224 - x.size()[3]

        x = F.pad(x, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])

        x = self.relu(x)
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        return x

class UNet(nn.Module):
    def __init__(self, args, bilinear=False):
        super(UNet, self).__init__()
        self.bilinear = bilinear
        self.args = args
        if self.args.dataset == 'miniImageNet' or self.args.dataset == 'cifar_fs':
            self.backbone = resnet18()
            self.num_features = 512
            self.num_cls = 100
            self.base_cls_num = 60
            self.inc_cls_num = 5
            self.sessions = 9
        else:
            if self.args.pretrain:
                self.backbone = resnet18(True)
            else:
                self.backbone = resnet18()
            self.num_features = 512
            self.num_cls = 200
            self.base_cls_num = 100
            self.inc_cls_num = 10
            self.sessions = 11
        
        self.slf_attn = MultiHeadAttention(1, 512, 512, 512, dropout=0.5)

        self.fc = nn.Linear(512, self.num_cls, bias=False)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        factor = 2 if bilinear else 1
        self.down = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=2),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
        self.up = nn.Sequential(
            nn.ConvTranspose2d(512, 512, kernel_size=3),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(512, 512, kernel_size=3, stride=2),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
        
        self.de_layer4 = Deconv(512, 256)
        self.de_layer3 = Deconv(256, 128)
        self.de_layer2 = Deconv(128, 64)
        self.de_final = final_conv()
        
        self.mode = 'avg'

    def encode(self, x):
        feat1, feat2, feat3, feat4 = self.backbone(x, return_mid=True)
        hidden_feat = self.down(feat4).squeeze(-1).squeeze(-1) # [-1, 512, 1, 1]
        if 'avg' in self.mode:
            return hidden_feat
        else:
            return feat1, feat2, feat3, feat4, hidden_feat

    def reconstruct(self, feat1, feat2, feat3, feat4, hidden_feat):
        rec4 = self.up(hidden_feat.unsqueeze(-1).unsqueeze(-1))# [-1, 512, 7, 7]

        rec3 = self.de_layer4(feat3, rec4) # [-1, 256, 14, 14]
        rec2 = self.de_layer3(feat2, rec3) # [-1, 128, 28, 28]
        rec1 = self.de_layer2(feat1, rec2) # [-1, 64, 56, 56]
        x = self.de_final(rec1)
        return x

    def forward(self, x):
        """
        feat1: [-1, 64, 56, 56]
        feat2: [-1, 128, 28, 28]
        feat3: [-1, 256, 14, 14]
        feat4: [-1, 512, 7, 7]
        """
        if 'avg' in self.mode:
            avg_feature = self.encode(x)
            return avg_feature
        else:
            feat1, feat2, feat3, feat4, avg_feat = self.encode(x)
            rec_image = self.reconstruct(feat1, feat2, feat3, feat4, avg_feat)
            loss_rec = 0.0
            loss_rec = F.mse_loss(x, rec_image)
            return loss_rec, avg_feat
    
    def forward_(self, proto, query, return_feature=False, return_proto=False):
        # support: [n_w, d]
        # query: [n_q, d]
        if proto.dim() < 3:
            n_w = proto.shape[0]
            proto = proto.unsqueeze(0).repeat(query.shape[0], 1, 1) # [n_q, n_w, d]
            query = query.unsqueeze(1) # [n_q, 1, d]
        else:
            n_w = proto.shape[1]
            query = query.unsqueeze(1) # [n_q, 1, d]
        if self.args.feature_imprinting:
            relation = F.normalize(proto, dim=-1, p=2) * F.normalize(query.expand_as(proto), dim=-1, p=2) # [n_q, n_w, d]
            proto = proto + relation * query.expand_as(proto) # 
        # Attn
        combined = torch.cat([proto, query], 1) # [n_q, n_w+1, d]
        combined = self.slf_attn(combined, combined, combined)
        proto, query = combined.split(n_w, 1)
        # MN(cos) PN(euc) RN(relation net)
        logits = F.cosine_similarity(query-torch.mean(query, dim=-1, keepdim=True), 
                                     proto-torch.mean(proto, dim=-1, keepdim=True),dim=-1) 
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
