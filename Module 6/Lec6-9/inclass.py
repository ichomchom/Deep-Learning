import torch


# sequence length is how long we expect the sequence to be,
# def attention_mask(seq_len, learned_embedding: torch.Tensor):
#     # learned_embedding: [embed_0, embed_1, ..., embed_n]
#     # embed_0 is floating point val, goin to be used from location x to a location x in attention to embedding
#     # embed_1 is go from a location x to 1 back
#     # embed_n is go from location x to location n back

#     embed = learned_embedding.new_full((
#         *learned_embedding.shape[:-1],
#         seq_len,
#         seq_len), float('-inf'))
#     # infinity matrix
#     # learned_embedding.new_full creates a new tensor on the device that embed lift on, using same datatype, fill it with negative inf

#     pos = torch.arange(seq_len)  # seq pos from 0 to seq length
#     # relative pos is position along 1 dim - the pos along other dim
#     rel_pos = pos[:, None] - pos[None, :]
#     # not all pos are valid, we want to mask out everything that beyond the length of our learned embedded will
#     valid_pos = (rel_pos >= 0) & (rel_pos < learned_embedding.shape[-1])
#     # valid pos are pos >= 0 or shorter than our learned embedded vector

#     # retrieve the corresponding rel pos after valid pos
#     embed[..., valid_pos] = learned_embedding[..., rel_pos[valid_pos]]
#     # retrieve the learned embedding from there then copy into embed

#     return embed

def causal_mask(size, rel_pos: torch.Tensor):
    if size > rel_pos.size(-1):
        rel_pos_padded = torch.cat(
            [
                rel_pos.new_full(
                    (
                        *rel_pos.shape[:-1],
                        size - rel_pos.size(-1),
                    ),
                    -float("inf"),
                ),
                rel_pos.flip(-1),
                rel_pos.new_full(
                    (
                        *rel_pos.shape[:-1],
                        size,
                    ),
                    -float("inf"),
                ),
            ],
            dim=-1,
        )
    else:
        rel_pos_padded = torch.cat(
            [
                rel_pos[..., :size].flip(-1),
                rel_pos.new_full(
                    (
                        *rel_pos.shape[:-1],
                        size,
                    ),
                    -float("inf"),
                ),
            ],
            dim=-1,
        )
    return rel_pos_padded[..., torch.arange(size)[None, :] - torch.arange(size)[:, None] + size - 1]


class TransformerLayer(torch.nn.Module):
    def __init__(self, embed_dim, num_heads, max_len=128):
        super().__init__()
        self.rel_pos = torch.nn.Parameter(torch.randn(num_heads, max_len))
        self.self_att = torch.nn.MultiheadAttention(
            embed_dim, num_heads, batch_first=True)
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(
                embed_dim, 4 * embed_dim), torch.nn.ReLU(), torch.nn.Linear(4 * embed_dim, embed_dim)
        )
        self.in_norm = torch.nn.LayerNorm(embed_dim)
        self.mlp_norm = torch.nn.LayerNorm(embed_dim)

    def forward(self, x):
        x_norm = self.in_norm(x)
        mask = causal_mask(x.size(1), self.rel_pos)
        mask = mask.repeat(x.size(0), 1, 1)
        x = x + self.self_att(x_norm, x_norm, x_norm, attn_mask=mask)[0]
        x = x + self.mlp(self.mlp_norm(x))
        return x


class Transformer(torch.nn.Module):
    def __init__(self, embed_dim, num_heads, num_layers):
        super().__init__()
        self.network = torch.nn.Sequential(
            torch.nn.Embedding(128, embed_dim),
            *[TransformerLayer(embed_dim, num_heads)
              for _ in range(num_layers)],
            torch.nn.Linear(embed_dim, 128),
        )

    def forward(self, x):
        return self.network(x)


def train():
    net = Transformer(256, 8, 4)
    net.cuda()

    with open(__file__, "rb") as f:
        code = f.read()
    data = torch.tensor([127] + list(code) + [0]).cuda()

    optim = torch.optim.AdamW(net.parameters(), lr=1e-3)

    for it in range(121):
        pred = net(data[None, :-1])[0]
        loss = torch.nn.functional.cross_entropy(pred, data[1:])
        optim.zero_grad()
        loss.backward()
        optim.step()
        if it % 10 == 0:
            print(f"loss = {float(loss)}")
            pred = net(data[None, :1])[0]
            print(data[1:2].cpu().detach().numpy())
            print(pred.argmax(-1).cpu().detach().numpy())
            pred = net(data[None, :10])[0]
            print(data[1:11].cpu().detach().numpy())
            print(pred.argmax(-1).cpu().detach().numpy())

    pred = net(data[None, :10])[0]
    print(data[1:11].cpu().detach().numpy())
    print(pred.argmax(-1).cpu().detach().numpy())
    torch.save(net, "transformer.pth")


def sample():
    import sys

    net = torch.load("transformer.pth", weights_only=False)
    # net.eval()
    data = [127]
    for _ in range(10000):
        pred = int(net(torch.tensor(data[-500:]).cuda()[None])[0, -1].argmax())
        sys.stdout.write(chr(pred))
        sys.stdout.flush()
        if pred == 0:
            break
        data.append(pred)


if __name__ == "__main__":
    train()
    sample()
