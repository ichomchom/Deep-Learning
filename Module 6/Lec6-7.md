```python
import torch
```


```python
class TransformerLayer(torch.nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()

        self.self_att = torch.nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(embed_dim, 4 * embed_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(4 * embed_dim, embed_dim)
        )
        self.in_norm = torch.nn.LayerNorm(embed_dim)
        self.mlp_norm = torch.nn.LayerNorm(embed_dim)

    def forward(self, x):
        x_norm = self.in_norm(x)
        x = x + self.self_att(x_norm, x_norm, x_norm)[0]
        x = x + self.mlp(self.mlp_norm(x))
        return x

# class Transformer(torch.nn.Module):
#     def __init__(self, embed_dim, num_heads, num_layers):
#         super().__init__()
#         self.layers = torch.nn.ModuleList(
#             [
#                 TransformerLayer(embed_dim, num_heads) for _ in range(num_layers)
#             ]
#         )

#     def forward(self, x):
#         for layer in self.layers:
#             x = layer(x)
#         return x

    ## Compact way
class Transformer(torch.nn.Module):
    def __init__(self, embed_dim, num_heads, num_layers):
        super().__init__()
        self.network = torch.nn.Sequential(
            *[
                TransformerLayer(embed_dim, num_heads) for _ in range(num_layers)
            ]
        )

    def forward(self, x):
        return self.network(x)

net = Transformer(128, 8, 4)
net(torch.rand(16, 10, 128)).shape
```




    torch.Size([16, 10, 128])




```python
net
```
