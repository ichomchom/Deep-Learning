```python
import torch
```


```python
torch_attention = torch.nn.MultiheadAttention(16, 4, 0, batch_first=True)
# 16 embeding dimension as input, 4 attentions, 0 dropout (disable), batch_first flag means that the first dimension of our inputs is patch size
```


```python
k, v, q = torch.rand(10, 3, 16), torch.rand(10, 3, 16), torch.rand(10, 5, 16)
o, _ = torch_attention(q, k, v)
print(o.shape)
print([(n, v.shape) for n, v in list(torch_attention.named_parameters())])
```

    torch.Size([10, 5, 16])
    [('in_proj_weight', torch.Size([48, 16])), ('in_proj_bias', torch.Size([48])), ('out_proj.weight', torch.Size([16, 16])), ('out_proj.bias', torch.Size([16]))]



```python
class MHA(torch.nn.Module):
    def __init__(self, embed_dim, num_heads) -> None:
        super().__init__()
        self.in_proj_k = torch.nn.Linear(embed_dim, embed_dim)
        self.in_proj_v = torch.nn.Linear(embed_dim, embed_dim)
        self.in_proj_q = torch.nn.Linear(embed_dim, embed_dim)
        self.out_proj = torch.nn.Linear(embed_dim, embed_dim)
        self.n_heads = num_heads

    def forward(self, q, k, v):
        from einops import rearrange
        p_q, p_k, p_v = self.in_proj_q(q), self.in_proj_k(k), self.in_proj_v(v)

        r_q = rearrange(p_q, 'b m (h d) -> b h m d', h=self.n_heads)
        # rearrange query, we have batch dimension, then the next dimension is the number of keys we have
        r_k = rearrange(p_k, 'b n (h d) -> b h n d', h=self.n_heads)
        r_v = rearrange(p_v, 'b n (h d) -> b h n d', h=self.n_heads)

        scores = torch.einsum('b h m d, b h n d -> b h m n', r_q, r_k)
        # einsum allows you to specify dimensions of inputs, then it will multiply the first and the second input together
        # element wise, and sum over everything that you didn't specify would be in the output
        # this give us the raw attention weight
        attn = torch.nn.functional.softmax(scores, dim=-1)
        result = torch.einsum('b h m n, b h n d -> b h m d', attn, r_v)

        r_result = rearrange(result, 'b h m d -> b m (h d)')

        return self.out_proj(r_result)

our_attention = MHA(16, 4)
o_our = our_attention(q, k, v)
o_torch, _ = torch_attention(q, k, v)
print(o.shape, o_torch.shape)
```

    torch.Size([10, 5, 16]) torch.Size([10, 5, 16])

