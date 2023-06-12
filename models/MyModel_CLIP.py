# import clip
from .CLIP import clip
import torch
import torch.nn as nn
from .ResNet import *
from .resnet20_cifar import resnet20
import torch.nn.functional as F

from .CLIP.simple_tokenizer import SimpleTokenizer as _Tokenizer

_tokenizer = _Tokenizer()


# from timm.models.layers import PatchEmbed

class MyModel(nn.Module):

    def __init__(self, 
                 dataset: str='cub_200', 
                 arch_name: str='ViT-B/32', 
                 prompt_len: int=4,
                 run_method: str='ours',
                 enable_text_prompt: bool=False,
                 text_prompt_len: int=4, 
                 class_names = None, 
                 version: str='V1'):
        super(MyModel, self).__init__()
        self.version = version
        self.arch_name = arch_name
        # initi incremental info
        if dataset == 'miniImageNet' or dataset == 'cifar_fs':
            self.num_cls, self.base_cls_num, self.inc_cls_num, self.sessions = 100, 60, 5, 9
            self.pretrained = False
        elif dataset == 'cub_200' or dataset == 'ImageNet_R':
            self.num_cls, self.base_cls_num, self.inc_cls_num, self.sessions = 200, 100, 10, 11
            self.pretrained = True
        else:
            raise Exception("Invalid dataset name {}".format(dataset))
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # init model
        if arch_name == 'ResNet18':
            # import pdb
            # pdb.set_trace()
            self.encoder = resnet18(self.pretrained)
            self.num_features = 512
        elif arch_name == 'ResNet20':
            self.encoder = resnet20()
            self.num_features = 64
        else: # otherwise all the CLIP
            self.encoder, self.preprocess = clip.load(arch_name, device="cpu")
            self.num_features = 512
        self.protos = nn.Linear(self.num_features, self.num_cls, bias=False)
        self.linear = nn.Linear(self.num_features, self.num_features, bias=False)
        self.linear_text = nn.Linear(self.num_features, self.num_features, bias=False)

        # used to store specific knowledge
        self.heads_vis = nn.ModuleList()
        self.heads_sem = nn.ModuleList()
        self.heads_vis.append(nn.Linear(self.num_features, self.base_cls_num, bias=False))
        self.heads_sem.append(nn.Linear(self.num_features, self.base_cls_num, bias=False))
        for i in range(self.sessions-1):
            self.heads_vis.append(nn.Linear(self.num_features, self.inc_cls_num, bias=False))
            self.heads_sem.append(nn.Linear(self.num_features, self.inc_cls_num, bias=False))
        
        # initialize visual prompts
        self.prompt = torch.randn((prompt_len, 768), requires_grad=True)
        self.prompt = nn.Parameter(self.prompt)
        self.prompt_len = prompt_len

        if enable_text_prompt:
            # modified from https://github.com/KaiyangZhou/CoOp/blob/main/trainers/coop.py
            self.class_names = class_names
            self.text_prompt_len = text_prompt_len
            self.InitTextPromptLearner()

        # for L2P
        self.run_method = run_method
        if run_method == 'L2P':
            embed_dim_prompt = 768
            embed_dim_key = 512
            self.M, self.N, self.Lp = 10, 5, 5
            self.freq = torch.zeros(self.M)
            self.prompt_pool = torch.randn((self.M, self.Lp, embed_dim_prompt), requires_grad=True)
            self.prompt_pool = nn.Parameter(self.prompt_pool)
            self.match_keys = torch.randn((self.M, embed_dim_key), requires_grad=True)
            self.match_keys = nn.Parameter(self.match_keys)
            self.total_prompt_len = self.N * self.Lp
            self.fc = nn.Linear(self.num_features, self.num_cls, bias=False)
            nn.init.uniform_(self.prompt_pool)
            nn.init.uniform_(self.match_keys)

    def InitTextPromptLearner(self):
        # self.n_ctx = self.text_prompt_len
        dtype = self.encoder.dtype
        # print("Initializing a generic context")
        # ctx_vectors = torch.empty(self.n_ctx, 512, dtype=dtype)
        # nn.init.normal_(ctx_vectors, std=0.02)

        # prompt_prefix = " ".join(["X"] * self.n_ctx)
        # self.ctx = nn.Parameter(ctx_vectors)  # to be optimized

        # self.class_names = [name.replace("_", " ") for name in self.class_names]
        # self.name_lens   = [len(_tokenizer.encode(name)) for name in self.class_names]

        # prompt_prefix = 'a photo of a {}, a type of bird.'
        # prompts = [prompt_prefix + " " + name + "." for name in self.class_names]
        # self.tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])

        # with torch.no_grad():
        #     embedding = self.encoder.token_embedding(self.tokenized_prompts).type(dtype)

        # self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        # self.register_buffer("token_suffix", embedding[:, 1 + self.n_ctx :, :])  # CLS, EOS


        prompts = [f'a photo of a {name}, a type of bird.' for name in self.class_names]
        self.tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])
        with torch.no_grad():
            self.token_embeddings = self.encoder.token_embedding(self.tokenized_prompts).type(dtype)


    # correspond to the pretrained embedding layer of L2P
    def PEL_encode(self, x, prompts=None, pos_emb='patch_cls'):
        if 'patch' in pos_emb:
            x = self.encoder.visual.conv1(x)  # shape = [*, width, grid, grid]
            x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
            x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        if 'cls' in pos_emb:
            x = torch.cat([self.encoder.visual.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]
            x = x + self.encoder.visual.positional_embedding.to(x.dtype)
            x = torch.cat((x, prompts.unsqueeze(0).repeat(x.shape[0],1 ,1)), dim=1) 
        return x

    # correspond to the pretrained transformer encoder of L2P
    def PTE_encode(self, x):
        x = self.encoder.visual.ln_pre(x)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.encoder.visual.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.encoder.visual.ln_post(x) # modified
        if self.encoder.visual.proj is not None:
            x = x @ self.encoder.visual.proj
        x = x[:, -self.total_prompt_len:].mean(dim=1)
        return x


    def encode_image(self, x: torch.Tensor, 
                    memory = None,
                    KPM = None,
                    gen_prompt=None,
                    gen_prompt_concat_layer=-1,
                    cap_layer: int=-1,
                    upd_layer: int=-1,
                    upd_targt: str='none',
                    enable_prompt:bool=False,
                    get_patch_embed:bool=False, 
                    return_all: bool=False,
                    linear: bool = False
                    ):
        if 'ViT' not in self.arch_name:
            x = self.encoder(x)
            x = self.avgpool(x).squeeze(-1).squeeze(-1)
            return x
        x = self.encoder.visual.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        if not get_patch_embed:

            x = torch.cat([self.encoder.visual.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]
            x = x + self.encoder.visual.positional_embedding.to(x.dtype)

            if gen_prompt is not None and gen_prompt_concat_layer == -1:
                x = torch.cat((x, gen_prompt.unsqueeze(0).repeat(x.shape[0],1 ,1)), dim=1) 
            else:
                if enable_prompt:
                    # concat behind last token
                    x = torch.cat((x, self.prompt.unsqueeze(0).repeat(x.shape[0],1 ,1)), dim=1) 
                    # concat before cls token
                    # x = torch.cat((self.prompt.unsqueeze(0).repeat(x.shape[0],1 ,1), x), dim=1)

            
            x = self.encoder.visual.ln_pre(x)

            x = x.permute(1, 0, 2)  # NLD -> LND
            if cap_layer != -1 and self.version=='V1':
                x = self.encoder.visual.transformer(x,
                                                    cap_layer=cap_layer,
                                                    gen_prompt=gen_prompt,
                                                    gen_prompt_concat_layer=gen_prompt_concat_layer,
                                                    prompt_len=self.prompt_len)
                return x
            else:
                x = self.encoder.visual.transformer(x,
                                                    memory=memory, 
                                                    KPM=KPM, 
                                                    cap_layer=cap_layer,
                                                    upd_layer=upd_layer,
                                                    upd_targt=upd_targt,
                                                    gen_prompt=gen_prompt,
                                                    gen_prompt_concat_layer=gen_prompt_concat_layer,
                                                    prompt_len=self.prompt_len,
                                                    version=self.version)

            x = x.permute(1, 0, 2)  # LND -> NLD
            x = self.encoder.visual.ln_post(x) # modified

            if self.encoder.visual.proj is not None:
                x = x @ self.encoder.visual.proj
            
            if not return_all:
                x = x[:, 0, :]
                # x = x[:, :self.prompt_len, :].mean(dim=1)
        if linear:
            x = self.linear(x)
        return x


    def encode_text(self, 
                    text=None, 
                    cls_list=None, 
                    num_seen_cls=200,
                    enable_text_prompt: bool=False):

        if enable_text_prompt:
            # import pdb
            # pdb.set_trace()
            dtype = self.encoder.dtype
            if cls_list is None:
                cls_list = torch.arange(num_seen_cls)

            # ctx = self.ctx.unsqueeze(0).expand(self.num_cls, -1, -1)
            # text_prompts =  prompts = torch.cat(
            #                                     [
            #                                         self.token_prefix[cls_list],  # (n_cls, 1, dim)
            #                                         ctx[cls_list],           # (n_cls, n_ctx, dim)
            #                                         self.token_suffix[cls_list],  # (n_cls, *, dim)
            #                                     ],
            #                                     dim=1,
            #                                 )
            # x = text_prompts + self.encoder.positional_embedding.type(self.encoder.dtype)
            
            x = self.token_embeddings[cls_list].to(self.encoder.positional_embedding.device) + self.encoder.positional_embedding.type(self.encoder.dtype)
            x = x.permute(1, 0, 2)  # NLD -> LND
            x = self.encoder.transformer(x)
            x = x.permute(1, 0, 2)  # LND -> NLD
            x = self.encoder.ln_final(x).type(self.encoder.dtype)

            x = x[torch.arange(x.shape[0]), self.tokenized_prompts[cls_list].argmax(dim=-1)] @ self.encoder.text_projection

            return x
        else:
            return self.encoder.encode_text(text)


    def forward(self, x, cur_session=0):
        """
        x: input data
        """
        feat_vis = self.encode_image(x)
        outs_vis = []
        for i in range(cur_session+1):
            outs_vis.append(
                F.linear(
                    F.normalize(feat_vis, p=2, dim=-1), F.normalize(self.heads_vis[i].weight, p=2, dim=-1)
                    ))
        outs_vis = torch.cat(outs_vis, dim=1)
        return outs_vis
    

    def forward_sem(self, feat_sem, cur_session=0):
        outs_sem = []
        for i in range(cur_session+1):
            outs_sem.append(
                F.linear(
                    F.normalize(feat_sem, p=2, dim=-1), F.normalize(self.heads_sem[i].weight, p=2, dim=-1)
                    ))
        outs_sem = torch.cat(outs_sem, dim=1)
        return outs_sem


