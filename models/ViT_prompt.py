import torch
import torch.nn as nn

class Prompt(nn.Module):
    def __init__(self, length=5, embed_dim=768, embedding_key='mean', prompt_init='uniform', prompt_pool=False, 
                 prompt_key=False, pool_size=None, top_k=None, batchwise_prompt=False, prompt_key_init='uniform',):
        super().__init__()

        self.length = length
        self.prompt_pool = prompt_pool
        self.embedding_key = embedding_key
        self.pool_size = pool_size
        self.top_k = top_k
        self.batchwise_prompt = batchwise_prompt

        if self.prompt_pool:
            if prompt_init == 'zero':
                self.prompt = nn.Parameter(torch.zeros(pool_size, length, embed_dim))
            elif prompt_init == 'uniform':
                self.prompt = nn.Parameter(torch.randn(pool_size, length, embed_dim))
                nn.init.uniform_(self.prompt)
        else:
            if prompt_init == 'zero':
                self.prompt = nn.Parameter(torch.zeros(length, embed_dim))
            elif prompt_init == 'uniform':
                self.prompt = nn.Parameter(torch.randn(length, embed_dim))
                nn.init.uniform_(self.prompt)

        
        # if using learnable prompt keys
        if prompt_key:
            key_shape = (pool_size, embed_dim)
            if prompt_key_init == 'zero':
                self.prompt_key = nn.Parameter(torch.zeros(key_shape))
            elif prompt_key_init == 'uniform':
                self.prompt_key = nn.Parameter(torch.randn(key_shape))
                nn.init.uniform_(self.prompt_key)
        else:
            # else use mean of prompt as key
            # only compatible with prompt, not prefix
            prompt_mean = torch.mean(self.prompt, dim=1)
            self.prompt_key = prompt_mean
    
    def l2_normalize(self, x, dim=None, epsilon=1e-12):
        """Normalizes a given vector or matrix."""
        square_sum = torch.sum(x ** 2, dim=dim, keepdim=True)
        x_inv_norm = torch.rsqrt(torch.maximum(square_sum, torch.tensor(epsilon, device=x.device)))
        return x * x_inv_norm
    
    def forward(self, x_embed, prompt_mask=None, cls_features=None):
        out = dict()
        if self.prompt_pool:
            if self.embedding_key == 'mean':
                x_embed_mean = torch.mean(x_embed, dim=1)
            elif self.embedding_key == 'max':
                x_embed_mean = torch.max(x_embed, dim=1)[0]
            elif self.embedding_key == 'mean_max':
                x_embed_mean = torch.max(x_embed, dim=1)[0] + 2 * torch.mean(x_embed, dim=1)
            elif self.embedding_key == 'cls':
                if cls_features is None:
                    x_embed_mean = torch.max(x_embed, dim=1)[0] # B, C
                else:
                    x_embed_mean = cls_features
            else:
                raise NotImplementedError("Not supported way of calculating embedding keys!")

            prompt_norm = self.l2_normalize(self.prompt_key, dim=1) # Pool_size, C
            x_embed_norm = self.l2_normalize(x_embed_mean, dim=1) # B, C

            similarity = torch.matmul(x_embed_norm, prompt_norm.t()) # B, Pool_size
            
            if prompt_mask is None:
                _, idx = torch.topk(similarity, k=self.top_k, dim=1) # B, top_k
                if self.batchwise_prompt:
                    prompt_id, id_counts = torch.unique(idx, return_counts=True, sorted=True)
                    # In jnp.unique, when the 'size' is specified and there are fewer than the indicated number of elements,
                    # the remaining elements will be filled with 'fill_value', the default is the minimum value along the specified dimension.
                    # Unless dimension is specified, this will be flattend if it is not already 1D.
                    if prompt_id.shape[0] < self.pool_size:
                        prompt_id = torch.cat([prompt_id, torch.full((self.pool_size - prompt_id.shape[0],), torch.min(idx.flatten()), device=prompt_id.device)])
                        id_counts = torch.cat([id_counts, torch.full((self.pool_size - id_counts.shape[0],), 0, device=id_counts.device)])
                    _, major_idx = torch.topk(id_counts, k=self.top_k) # top_k
                    major_prompt_id = prompt_id[major_idx] # top_k
                    # expand to batch
                    idx = major_prompt_id.expand(x_embed.shape[0], -1) # B, top_k
            else:
                idx = prompt_mask # B, top_k
            
            batched_prompt_raw = torch.gather(self.prompt.expand(idx.shape[0], -1, -1, -1), 
                                                index=idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, self.length, x_embed.shape[-1]), dim=1) # B, top_k, length, C
            batch_size, top_k, length, c = batched_prompt_raw.shape
            batched_prompt = batched_prompt_raw.reshape(batch_size, top_k * length, c) # B, top_k * length, C

            out['prompt_idx'] = idx

            # Debugging, return sim as well
            out['prompt_norm'] = prompt_norm
            out['x_embed_norm'] = x_embed_norm
            out['similarity'] = similarity

            # Put pull_constraint loss calculation inside
            batched_key_norm = torch.gather(prompt_norm.unsqueeze(0).expand(batch_size, -1, -1), 
                                            index=idx.unsqueeze(-1).expand(-1, -1, c), dim=1) # B, top_k, C
            out['selected_key'] = batched_key_norm
            x_embed_norm = x_embed_norm.unsqueeze(1) # B, 1, C
            sim = batched_key_norm * x_embed_norm # B, top_k, C
            reduce_sim = torch.sum(sim) / x_embed.shape[0] # Scalar

            out['reduce_sim'] = reduce_sim
        else:
            batched_prompt = self.prompt.unsqueeze(0).expand(x_embed.shape[0], -1, -1)
        
        # The input with the prompt concatenated to the front. [B, prompt+token, C]
        out['total_prompt_len'] = batched_prompt.shape[1]
        out['prompted_embedding'] = torch.cat([batched_prompt, x_embed], dim=1)

        return out


class PromptSplit(nn.Module):
    def __init__(self, length=5, embed_dim=768, embedding_key='mean', prompt_init='uniform', prompt_pool=False, 
                 prompt_key=False, pool_size_old=None, pool_size_new=None, top_k=None, batchwise_prompt=False, prompt_key_init='uniform',):
        super().__init__()
        
        self.length = length
        self.prompt_pool = prompt_pool
        self.embedding_key = embedding_key
        self.pool_size_old = pool_size_old
        self.pool_size_new = pool_size_new
        self.top_k = top_k
        self.batchwise_prompt = batchwise_prompt

        if self.prompt_pool:
            if prompt_init == 'zero':
                self.prompt_old = nn.Parameter(torch.zeros(pool_size_old, length, embed_dim))
                self.prompt_new = nn.Parameter(torch.zeros(pool_size_new, length, embed_dim))
            elif prompt_init == 'uniform':
                self.prompt_old = nn.Parameter(torch.randn(pool_size_old, length, embed_dim))
                self.prompt_new = nn.Parameter(torch.randn(pool_size_new, length, embed_dim))
                nn.init.uniform_(self.prompt_old)
                nn.init.uniform_(self.prompt_new)
        else:
            if prompt_init == 'zero':
                self.prompt = nn.Parameter(torch.zeros(length, embed_dim))
            elif prompt_init == 'uniform':
                self.prompt = nn.Parameter(torch.randn(length, embed_dim))
                nn.init.uniform_(self.prompt)
        
        # if using learnable prompt keys
        if prompt_key:
            key_shape_old = (pool_size_old, embed_dim)
            key_shape_new = (pool_size_new, embed_dim)
            if prompt_key_init == 'zero':
                self.prompt_key_old = nn.Parameter(torch.zeros(key_shape_old))
                self.prompt_key_new = nn.Parameter(torch.zeros(key_shape_new))
            elif prompt_key_init == 'uniform':
                self.prompt_key_old = nn.Parameter(torch.randn(key_shape_old))
                self.prompt_key_new = nn.Parameter(torch.randn(key_shape_new))
                nn.init.uniform_(self.prompt_key_old)
                nn.init.uniform_(self.prompt_key_new)
        else:
            # else use mean of prompt as key
            # only compatible with prompt, not prefix
            prompt_mean = torch.mean(self.prompt, dim=1)
            self.prompt_key = prompt_mean
    
    @property
    def prompt(self):
        return torch.cat([self.prompt_old, self.prompt_new])

    @property
    def prompt_key(self):
        return torch.cat([self.prompt_key_old, self.prompt_key_new])
    
    @property
    def pool_size(self):
        return self.pool_size_old + self.pool_size_new

    def l2_normalize(self, x, dim=None, epsilon=1e-12):
        """Normalizes a given vector or matrix."""
        square_sum = torch.sum(x ** 2, dim=dim, keepdim=True)
        x_inv_norm = torch.rsqrt(torch.maximum(square_sum, torch.tensor(epsilon, device=x.device)))
        return x * x_inv_norm
    
    def forward(self, x_embed, prompt_mask=None, cls_features=None):
        out = dict()
        if self.prompt_pool:
            if self.embedding_key == 'mean':
                x_embed_mean = torch.mean(x_embed, dim=1)
            elif self.embedding_key == 'max':
                x_embed_mean = torch.max(x_embed, dim=1)[0]
            elif self.embedding_key == 'mean_max':
                x_embed_mean = torch.max(x_embed, dim=1)[0] + 2 * torch.mean(x_embed, dim=1)
            elif self.embedding_key == 'cls':
                if cls_features is None:
                    x_embed_mean = torch.max(x_embed, dim=1)[0] # B, C
                else:
                    x_embed_mean = cls_features
            else:
                raise NotImplementedError("Not supported way of calculating embedding keys!")

            prompt_norm = self.l2_normalize(self.prompt_key, dim=1) # Pool_size, C
            x_embed_norm = self.l2_normalize(x_embed_mean, dim=1) # B, C

            similarity = torch.matmul(x_embed_norm, prompt_norm.t()) # B, Pool_size
            
            if prompt_mask is None:
                _, idx = torch.topk(similarity, k=self.top_k, dim=1) # B, top_k
                if self.batchwise_prompt:
                    prompt_id, id_counts = torch.unique(idx, return_counts=True, sorted=True)
                    # In jnp.unique, when the 'size' is specified and there are fewer than the indicated number of elements,
                    # the remaining elements will be filled with 'fill_value', the default is the minimum value along the specified dimension.
                    # Unless dimension is specified, this will be flattend if it is not already 1D.
                    if prompt_id.shape[0] < self.pool_size:
                        prompt_id = torch.cat([prompt_id, torch.full((self.pool_size - prompt_id.shape[0],), torch.min(idx.flatten()), device=prompt_id.device)])
                        id_counts = torch.cat([id_counts, torch.full((self.pool_size - id_counts.shape[0],), 0, device=id_counts.device)])
                    _, major_idx = torch.topk(id_counts, k=self.top_k) # top_k
                    major_prompt_id = prompt_id[major_idx] # top_k
                    # expand to batch
                    idx = major_prompt_id.expand(x_embed.shape[0], -1) # B, top_k
            else:
                idx = prompt_mask # B, top_k
            
            batched_prompt_raw = torch.gather(self.prompt.expand(idx.shape[0], -1, -1, -1), 
                                                index=idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, self.length, x_embed.shape[-1]), dim=1) # B, top_k, length, C
            batch_size, top_k, length, c = batched_prompt_raw.shape
            batched_prompt = batched_prompt_raw.reshape(batch_size, top_k * length, c) # B, top_k * length, C

            out['prompt_idx'] = idx

            # Debugging, return sim as well
            out['prompt_norm'] = prompt_norm
            out['x_embed_norm'] = x_embed_norm
            out['similarity'] = similarity

            # Put pull_constraint loss calculation inside
            batched_key_norm = torch.gather(prompt_norm.unsqueeze(0).expand(batch_size, -1, -1), 
                                            index=idx.unsqueeze(-1).expand(-1, -1, c), dim=1) # B, top_k, C
            out['selected_key'] = batched_key_norm
            x_embed_norm = x_embed_norm.unsqueeze(1) # B, 1, C
            sim = batched_key_norm * x_embed_norm # B, top_k, C
            reduce_sim = torch.sum(sim) / x_embed.shape[0] # Scalar

            out['reduce_sim'] = reduce_sim
        else:
            batched_prompt = self.prompt.unsqueeze(0).expand(x_embed.shape[0], -1, -1)
        
        # The input with the prompt concatenated to the front. [B, prompt+token, C]
        out['total_prompt_len'] = batched_prompt.shape[1]
        out['prompted_embedding'] = torch.cat([batched_prompt, x_embed], dim=1)

        return out