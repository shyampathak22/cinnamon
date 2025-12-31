import torch
import torch.nn as nn
import torch.nn.functional as F

class RoPE(nn.Module):

    def __init__(self, d_k, max_seq_len, base=10000):
        super().__init__()
        # calcualte inverse frequency, and save it as a buffer
        self.register_buffer('theta', 1 / (base**(2*(torch.arange(0, d_k//2)) / d_k)))

        # create range of pos indices
        self.pos = torch.arange(max_seq_len)

        # precompute and cache angles for efficiency
        angles = torch.outer(self.pos, self.theta).unsqueeze(0).unsqueeze(2)
        self.register_buffer('cos_cache', angles.cos())
        self.register_buffer('sin_cache', angles.sin())

    def forward(self, x):
        # split seq and apply rotations using corresponsing cache slice
        x1, x2 = torch.chunk(x, 2, dim=-1)
        x1_rot = x1 * self.cos_cache[:, :x1.size(1)] - x2 * self.sin_cache[:, :x2.size(1)]
        x2_rot = x1 * self.sin_cache[:, :x1.size(1)] + x2 * self.cos_cache[:, :x2.size(1)]

        # concatenate and return
        return torch.concat((x1_rot, x2_rot), dim=-1)
    
class PoPE(nn.Module):

    def __init__(self, d_k, max_seq_len, base=10000):
        super().__init__()
        # calcualte inverse frequency, and save it as a buffer
        self.register_buffer('theta', 1 / (base**(2*(torch.arange(0, d_k//2)) / d_k)))

        # create range of pos indices
        self.register_buffer('pos', torch.arange(max_seq_len))

        self.delta = nn.Parameter(torch.zeros(d_k // 2))

    def forward(self, x):
        angles = torch.outer(self.pos, self.theta).unsqueeze(0).unsqueeze(2)
        cos = (angles + self.delta).cos()
        sin = (angles + self.delta).sin()
        mu1, mu2 = torch.chunk(F.softplus(x), 2, dim=-1)
        x1 = mu1 * cos[:, :mu1.size(1)] - mu2 * sin[:, :mu2.size(1)]
        x2 = mu1 * sin[:, :mu1.size(1)] + mu2 * cos[:, :mu2.size(1)]
        return torch.concat((x1, x2), dim=-1)

class MultiHeadAttention(nn.Module):

    def __init__(self, d_model, n_heads, max_seq_len, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        # Q, K, V, and output projections
        self.q = nn.Linear(d_model, d_model, bias=False)
        self.k = nn.Linear(d_model, d_model, bias=False)
        self.v = nn.Linear(d_model, d_model, bias=False)
        self.o = nn.Linear(d_model, d_model, bias=False)

        # self.dropout = nn.Dropout(p=dropout)
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.max_seq_len = max_seq_len

        # init RoPE class
        self.rope = RoPE(self.d_k, self.max_seq_len)

    # def sdpa(self, query, key, value, attn_mask="causal"):
    #     # calculate Q @ K^T  scores, apply scale to prevent dot products from becoming too large in high dims
    #     # Q @ K^T scores represent the similarity between each token and all other tokens in the sequence
    #     scores = (query @ key.transpose(-2, -1)) / self.d_k**0.5

    #     # apply causal attention mask using upper traingualr matrix, prevent looking ahead in seqeuence
    #     if attn_mask == "causal":
    #         scores = scores.masked_fill(torch.triu(torch.ones_like(scores), diagonal=1).bool(), float('-inf'))

    #     # calculate softmax and apply dropout
    #     attn_qk = F.softmax(scores, dim=-1)
    #     attn_qk = self.dropout(attn_qk)

    #     # calculate QK^T @ V scores
    #     # this weighs output information (V) based on the similarity score calculated above (Q @ K^T)
    #     attn = attn_qk @ value
    #     return attn
    
    def forward(self, query, key, value, attn_mask=None):
        # calculate batch size by finding number of queries
        batch_size = query.size(0)

        #apply rope and projections
        query = self.rope(self.q(query).view(batch_size, -1, self.n_heads, self.d_k)).transpose(1, 2)
        key = self.rope(self.k(key).view(batch_size, -1, self.n_heads, self.d_k)).transpose(1, 2)
        value = self.v(value).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)

        # pass to SPDA and apply output projection
        attn_o = F.scaled_dot_product_attention(query, key, value, is_causal=True).transpose(1, 2).reshape(batch_size, -1, self.d_k*self.n_heads)
        out = self.o(attn_o)
        return out
    
class MultiheadLatentAttention(nn.Module):
    def __init__(self, d_model, d_ckv, d_cq, n_head, d_head, d_rope, max_seq_len):
        super().__init__()
        self.n_head = n_head
        self.d_head = d_head
        self.w_dkv = nn.Linear(d_model, d_ckv, bias=False)
        self.w_uk = nn.Linear(d_ckv, d_head*n_head, bias=False)
        self.w_uv = nn.Linear(d_ckv, d_head*n_head, bias=False)
        self.w_dq = nn.Linear(d_model, d_cq, bias=False)
        self.w_uq = nn.Linear(d_cq, d_head*n_head, bias=False)
        self.d_rope = d_rope
        self.w_qr = nn.Linear(d_cq, d_rope*n_head, bias=False)
        self.w_kr = nn.Linear(d_model, d_rope, bias=False)
        self.pope = PoPE(d_rope, max_seq_len)
        self.w_out = nn.Linear(d_head*n_head, d_model, bias=False)

    def forward(self, x):
        batch_size = x.size(0)

        c_kv = self.w_dkv(x)
        key_c = self.w_uk(c_kv).view(batch_size, -1, self.n_head, self.d_head).transpose(1, 2)
        value = self.w_uv(c_kv).view(batch_size, -1, self.n_head, self.d_head).transpose(1, 2)
        c_q = self.w_dq(x)
        query_c = self.w_uq(c_q).view(batch_size, -1, self.n_head, self.d_head).transpose(1, 2)

        query_r = self.pope(self.w_qr(c_q).view(batch_size, -1, self.n_head, self.d_rope)).transpose(1, 2)
        key_r = self.pope(self.w_kr(x).unsqueeze(2)).transpose(1, 2).expand(-1, self.n_head, -1, -1)
        query = torch.concatenate((query_c, query_r), dim=-1)
        key = torch.concatenate((key_c, key_r), dim=-1)

        attn_o = F.scaled_dot_product_attention(query, key, value, is_causal=True, scale=(self.d_head + self.d_rope)**-0.5).transpose(1, 2).reshape(batch_size, -1, self.d_head*self.n_head)
        return self.w_out(attn_o)








    
if __name__ == "__main__":
    vocab_size = 50257
    d_model = 512
    max_seq_len = 1024
    batch_size = 32
    n_heads = 8
    attn = MultiHeadAttention(d_model, n_heads, max_seq_len)
    x = torch.randn(batch_size, max_seq_len, d_model)
    out= attn(x, x, x)
    print(f"output shape: {out.shape}")
    mla = MultiheadLatentAttention(
      d_model=512, d_ckv=256, d_cq=256,
      n_head=8, d_head=64, d_rope=32, max_seq_len=1024
    )
    x = torch.randn(4, 128, 512)
    out = mla(x)
    print(f"mla output shape: {out.shape}")
