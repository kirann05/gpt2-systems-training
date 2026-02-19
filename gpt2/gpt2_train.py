from dataclasses import dataclass
# pip install torch==2.0.1+cu117 torchvision==0.15.2+cu117 torchaudio==2.0.2+cu117 -f https://download.pytorch.org/whl/torch_stable.html
import torch
import torch.nn as nn
from torch.nn import functional as F
import math
import inspect
import dataloader

class CausalSelfAttention(nn.Module):

    def __init__(self, config, sliding_window = False, grouped_query = False ):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # batched q,k,v projections 
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1
        # regularization
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.sliding_window = sliding_window
        self.grouped_query = grouped_query
        self.window_size = config.window_size

        # grouped query attention with kv-cache
        # if self.grouped_query: 
        #     self.n_kv_heads = config.n_head if config.n_kv_heads is None else config.n_kv_heads # number of heads for key and value
        #     self.n_head_q = config.n_head # number of heads for queries
        #     self.n_rep = self.n_head_q // self.n_kv_heads # no of times heads of k,v needs to be repeated to match q heads
        #     self.head_dim = self.n_embd // self.n_head # dim of each head

        #     self.linear_q = nn.Linear(self.n_embd, self.n_embd) # linear projection of queries
        #     self.linear_kv = nn.Linear(self.n_embd , 2 * self.n_kv_heads * self.head_dim) # k,v linear projection 

        #     self.k_cache = torch.zeros((config.batch_size, config.seq_length, self.n_kv_heads, self.head_dim))
        #     self.v_cache = torch.zeros((config.batch_size, config.seq_length, self.n_kv_heads, self.head_dim))



        if self.sliding_window: 
            window_mask = torch.triu(torch.ones(config.block_size, config.block_size), 1-self.window_size) # upper triangle
            casual_mask = torch.tril(torch.ones(config.block_size, config.block_size)) # lower triangle
            combined_mask = casual_mask - window_mask
            # not really a 'bias', more of a mask, but following the OpenAI/HF naming though
            self.register_buffer("bias", combined_mask.view(1,1,config.block_size, config.block_size))
        else:
            # not really a 'bias', more of a mask, but following the OpenAI/HF naming though
            self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                     .view(1, 1, config.block_size, config.block_size))

    def forward(self, x, start_pos = None, freq = None):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality
        
        # if self.grouped_query: 
        #     q = self.linear_q(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        #     kv = self.linear_kv(x).view(B, T, 2, self.n_kv_heads, self.head_dim ) #(4, 1024, 512) (4, 1024, 2, 4, 64)
        #     k, v = kv.unbind(2) # k (4, 1024, 4, 64)

        #     k = k.transpose(1, 2) # (4, 4, 1024, 64)
        #     v = v.transpose(1, 2)

        #     self.k_cache[:B, start_pos:start_pos+ T] = k
        #     self.v_cache[:B, start_pos:start_pos+ T] = v

        #     k = self.k_cache[:B, :start_pos+T]
        #     v = self.v_cache[:B, :start_pos+T]

        #     k = k.repeat_interleave(self.n_rep, dim = 1)
        #     v = v.repeat_interleave(self.n_rep, dim = 1)


        # else:
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        # y = F.scaled_dot_product_attention(q, k, v, is_causal=True) # flash attention

        # you can use below 4 lines instead of flash attention by pytorch. 
        # but the memory to gpu kernel read/write times decreases significanlty using flash attention. 
        # also, the online softmax calculation is performed in flash attention that makes the computation much faster compared to normal softmax
        # achieving 5000ms /step, previously 11k ms/step time.

        att = (q @ k.transpose(-2,-1)) * (1.0/ math.sqrt(k.size(-1)))
        if not self.sliding_window: 
            att = att.masked_fill(self.bias[:,:,:T,:T] ==0 , float('-inf'))
        else: 
            att = att.masked_fill(self.bias[:,:,:T,:T] == -1, float('-inf'))
            att = att.masked_fill(self.bias[:,:,:T,:T] == 1, float('-inf'))
        
        att = F.softmax(att, dim=-1)
        y = att @ v
        # .contiguous() stores a copy of tensor in the memory.
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side.
        # output projection
        y = self.c_proj(y)
        return y

class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu    = nn.GELU(approximate='tanh') # there are two versions of gelu, paper uses approximation of tanh. 
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x

class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config, sliding_window=True)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        '''
        from original transformer architecture, GPT2 differs in architecture changes
        1. first goes through layer norm and then into the attention block. 
        2. first goes through layer norm and them into the MLP
        3. in addition to these changes, a layer norm is added at the end of the architecture.
        '''
        x = x + self.attn(self.ln_1(x)) # attention is where tokens communicate each other - reduced operation
        x = x + self.mlp(self.ln_2(x)) # no communication is performed here. - mapping operation 
        # hence this is kind of map-reduce operation. 
        return x

@dataclass
class GPTConfig:
    block_size: int = 1024 # max sequence length
    vocab_size: int = 50257 # number of tokens: 50,000 BPE merges + 256 bytes tokens + 1 <|endoftext|> token
    n_layer: int = 12 # number of layers
    n_head: int = 12 # number of heads
    n_embd: int = 768 # embedding dimension
    window_size: int = 128 # window size for sliding window attention 
    n_kv_heads: int = 4 # number of heads for key, value 
    batch_size: int = 4 # max batch size
    seq_length: int = 1024 # max sequence length 

class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd), # weights of token embeddings 
            wpe = nn.Embedding(config.block_size, config.n_embd), # weights of positional embeddings 
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # weight sharing scheme
        self.transformer.wte.weight = self.lm_head.weight

        # init params
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, 'NANOGPT_SCALE_INIT'):
                std *= (2 * self.config.n_layer) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets = None):
        # idx is of shape (B, T)
        B, T = idx.size()
        assert T <= self.config.block_size, f"Cannot forward sequence of length {T}, block size is only {self.config.block_size}"
        # forward the token and posisition embeddings
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device) # shape (T)
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (T, n_embd)
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (B, T, n_embd)
        x = tok_emb + pos_emb
        # forward the blocks of the transformer
        for block in self.transformer.h:
            x = block(x)
        # forward the final layernorm and the classifier
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x) # (B, T, vocab_size)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    @classmethod
    def from_pretrained(cls, model_type):
        """Loads pretrained GPT-2 model weights from huggingface"""
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        from transformers import GPT2LMHeadModel
        print("loading weights from pretrained gpt: %s" % model_type)

        # n_layer, n_head and n_embd are determined from model_type
        config_args = {
            'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
            'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024), # 350M params
            'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280), # 774M params
            'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600), # 1558M params
        }[model_type]
        config_args['vocab_size'] = 50257 # always 50257 for GPT model checkpoints
        config_args['block_size'] = 1024 # always 1024 for GPT model checkpoints
        # create a from-scratch initialized minGPT model
        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')] # discard this mask / buffer, not a param

        # init a huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')] # ignore these, just a buffer
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')] # same, just the mask (buffer)
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them
        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model
    
    def configureOptimizer(self, weight_decay, learning_rate, device):
        param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups, any param that is 2D will be decayed esle no
        # which is all weight tensors in matmul + embd decays, all bias and layer norms dont. 
        decay_params = [p for n, p in param_dict.items() if p.dim() >=2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]

        optim_groups = [{'params': decay_params, 'weight_decay':weight_decay, 
                        'params': nodecay_params, 'weight_decay': 0.0}]

        # kernel fusion - implimenting this, down to 2k ms/steps time.
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and 'cuda' in device

        optimizer = torch.optim.AdamW(optim_groups, lr = learning_rate, fused= use_fused, betas = (0.9, 0.95), eps = 1e-8)
        return optimizer

# pip install tiktoken
import tiktoken
import os
import numpy as np
def load_tokens(filename): 
    npt = np.load(filename)
    npt = npt.astype(np.float32)
    ppt = torch.tensor(npt, dtype = torch.long)
    return ppt

class DataLoader:
    def __init__(self, B, T, split):
        self.B =B
        self.T = T
        assert split in {'train', 'val'}

        data_root = "edu_fineweb10B"
        shards = os.listdir(data_root)
        shards = [s for s in shards if split in s]
        shards = sorted(shards)
        shards = [os.path.join(data_root, s) for s in shards]
        self.shards = shards

        self.current_shard = 0
        self.tokens = load_tokens(self.shards[self.current_shard])
        self.current_pos = self.B * self.T 

        # with open('input.txt', 'r') as f:
        #     text = f.read()
        # enc = tiktoken.get_encoding('gpt2')    
        # tokens = enc.encode(text)
        # self.tokens = torch.tensor(tokens)

        # self.current_pos = 0
        self.reset()

    def reset(self): 
        self.current_shard = 0
        self.tokens = load_tokens(self.shards[self.current_shard])
        self.current_pos = self.B * self.T

    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.current_pos: self.current_pos+B*T+1]
        x = buf[:-1].view(B,T)
        y = buf[1:].view(B,T)

        self.current_pos += B*T
        if self.current_pos + (B*T+1) > len(self.tokens):
            self.current_shard = (self.current_shard +1) % len(self.shards)
            self.tokens = load_tokens(self.shards[self.current_shard])
            self.current_pos = B* T

        return x, y
        


device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = "mps"
print(f"using device: {device}") 

torch.manual_seed(1337)
if torch.cuda.is_available():
    torch.cuda.manual_seed(1337)

# gradient accumulation for bigger batch sizes: 
# commenting them, because my gpu takes much time to perform gradient acuumulaiton. 
# total_batch_size = 524288
# B = 4
# T = 1024
# assert total_batch_size % (B*T) == 0 
# grad_accum_steps = total_batch_size // (B*T)

B = 4 # batch size
T = 1024 # token length
train_loader = dataloader.DataLoader(B, T, 'train')
val_loader = dataloader.DataLoader(B, T, 'val')
# uses tffloat 32 for all other layers that are not scale downed by autocast. this is a bit faster than float32 precision bits. 
# this reduced the time for each step from 20k ms to 15k ms
torch.set_float32_matmul_precision('high')

# model = GPT.from_pretrained('gpt2')
model = GPT(GPTConfig(vocab_size = 50304)) # random model initilaization, making the vocab_size power of 2, to make computation faster. 
# increasing the vocab size to 50304: 3k-4k ms / step from 5k-6k ms/step time. 
model.to(device)

# in general the python interpreter executes in eager mode- goes through each layer one by one
# decreases the read-write operations between memory and kernels.
# compiler here takes the entire architecture as an object and performs training, hence the increase in speed. 
# require python 3.10-dev version to run torch.compile, not compatible with latest version of python. 
# torch.compile is not compatible with windows, you need linux or WSL on windows machine.
model = torch.compile(model)

# lr scheduler from gpt3 paper.
import math
max_lr = 6e-4
min_lr = max_lr * 0.1
warmup_steps = 715
max_steps = 19073
def get_lr(it):
    if it < warmup_steps:
        return max_lr * (it+1) / warmup_steps
    if it > max_steps:
        return min_lr
    decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5* (1.0+math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)

import time

# commenting this for implimenting weight decay
# optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, betas = (0.9, 0.95), eps = 1e-8) # betas and eps from gpt-3 paper.

optimizer = model.configureOptimizer(weight_decay = 0.1, learning_rate = 6e-4, device= device)
for step in range(max_steps): 
    t0 = time.time()

    if step % 100 ==0: 
        model.eval()
        val_loader.reset()
        with torch.no_grad():
            val_loss_accum = 0.0
            val_loss_steps = 100
            for _ in range(val_loss_steps):
                x, y = val_loader.next_batch()
                x = torch.from_numpy(x)
                y = torch.from_numpy(y)
                x, y = x.to(device), y.to(device)
                with torch.autocast(device_type=device, dtype = torch.bfloat16):
                    logits, loss = model(x,y)
                loss = loss / val_loss_steps
                val_loss_accum += loss.detach()
            print(f"validation loss: {val_loss_accum:.4f}")
    
    # training loop
    model.train()
    x, y = train_loader.next_batch()
    x ,y = x.to(device), y.to(device)
    optimizer.zero_grad()
    with torch.autocast(device_type= device, dtype=torch.bfloat16):
        logits, loss = model(x, y)

    loss.backward()

    # commenting them, because my gpu takes much time to perform gradient acuumulaiton. 
    # for micro_step in range(grad_accum_steps):
    #     x, y = train_loader.next_batch()
    #     x ,y = x.to(device), y.to(device)
    #     # this make some of the layers like MLP into float 16 bit. runs much faster. 
    #     # this reduced the time for each step from 15k ms to 11k ms
    #     with torch.autocast(device_type= device, dtype=torch.bfloat16):
    #         logits, loss = model(x, y)
    #     loss = loss / grad_accum_steps
    #     loss.backward()

    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) # from gpt 3 paper. clip to mo more than 1.0

    #cosine learning rate
    lr = get_lr(step)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr # set the lr manually, cosine lr ends here.

    optimizer.step()
    torch.cuda.synchronize()
    t1 = time.time()
    dt = (t1 - t0)*1000
    tokens_per_sec = (train_loader.B * train_loader.T) / (t1-t0)
    print(f"step {step} | loss: {loss.item():.6f} | lr: {lr} | norm: {norm:.4f} | dt: {dt:.2f}ms, | tokens/sec: {tokens_per_sec:.2f}")


# torch.manual_seed(42)
# torch.cuda.manual_seed(42)
# while x.size(1) < max_length: 
#     with torch.no_grad():
#         logits = model(x)
#         logits = logits[:, -1, :]
#         probs = F.softmax(logits, dim=-1)
#         topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
#         ix = torch.multinomial(topk_probs, 1)
#         xcol = torch.gather(topk_indices, -1, ix)
#         x = torch.cat((x, xcol), dim=1)

# for i in range(max_return_seq):
#     tokens = x[i, :max_length].tolist()
#     decoded = enc.decode(tokens)
#     print(">", decoded)


