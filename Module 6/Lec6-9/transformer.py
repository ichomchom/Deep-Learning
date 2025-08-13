import torch


# sequence length is how long we expect the sequence to be,
def attention_mask(seq_len, learned_embedding: torch.Tensor):
    # learned_embedding: [embed_0, embed_1, ..., embed_n]
    # embed_0 is floating point val, goin to be used from location x to a location x in attention to embedding
    # embed_1 is go from a location x to 1 back
    # embed_n is go from location x to location n back

    embed = learned_embedding.new_full((
        *learned_embedding.shape[:-1],
        seq_len,
        seq_len), float('-inf'))
    # infinity matrix
    # learned_embedding.new_full creates a new tensor on the device that embed lift on, using same datatype, fill it with negative inf

    pos = torch.arange(seq_len)  # seq pos from 0 to seq length
    # relative pos is position along 1 dim - the pos along other dim
    rel_pos = pos[:, None] - pos[None, :]
    # not all pos are valid, we want to mask out everything that beyond the length of our learned embedded will
    valid_pos = (rel_pos >= 0) & (rel_pos < learned_embedding.shape[-1])
    # valid pos are pos >= 0 or shorter than our learned embedded vector

    # retrieve the corresponding rel pos after valid pos
    embed[..., valid_pos] = learned_embedding[..., rel_pos[valid_pos]]
    # retrieve the learned embedding from there then copy into embed

    return embed


class TransformerLayer(torch.nn.Module):
    def __init__(self, embed_dim, num_heads, max_len=512):
        super().__init__()

        self.rel_pos = torch.nn.Parameter(torch.randn(num_heads, max_len))
        self.self_att = torch.nn.MultiheadAttention(
            embed_dim, num_heads, batch_first=True)
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(embed_dim, 4 * embed_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(4 * embed_dim, embed_dim)
        )
        self.in_norm = torch.nn.LayerNorm(embed_dim)
        self.mlp_norm = torch.nn.LayerNorm(embed_dim)

    def forward(self, x):
        x_norm = self.in_norm(x)
        mask = attention_mask(x.size(1), self.rel_pos)
        # attention mask allow to feed a floating value is added to attention before the softmax
        mask = mask.repeat(x.size(0), 1, 1)

        x = x + self.self_att(x_norm, x_norm, x_norm,
                              attn_mask=mask)[0]
        x = x + self.mlp(self.mlp_norm(x))
        return x


class Transformer(torch.nn.Module):
    def __init__(self, embed_dim, num_heads, num_layers):
        super().__init__()
        self.network = torch.nn.Sequential(
            torch.nn.Embedding(128, embed_dim),
            *[
                TransformerLayer(embed_dim, num_heads) for _ in range(num_layers)
            ],
            torch.nn.Linear(embed_dim, 128),
        )

    def forward(self, x):
        return self.network(x)


def train():
    # read the source code
    with open(__file__) as f:
        code = f.read()
    # tokenize the source code as just characters
    # characters characters prediction after source code of this file
    tokens = torch.as_tensor([127] + [ord(c) for c in code] + [0]).cuda()
    # 2 special tokens for beginning of file and the end of file
    # beginning of file is 127 which is unused
    # end of file is 0

    net = Transformer(256, 8, 4)
    net.cuda()
    optim = torch.optim.AdamW(net.parameters(), lr=0.001)

    for it in range(121):
        # predict certain output for every token in the sequence, except the last one, last one is EOS
        pred = net(tokens[None, :-1])[0]
        loss = torch.nn.functional.cross_entropy(
            pred, tokens[1:])  # not predict the first token

        optim.zero_grad()
        loss.backward()
        optim.step()
        if it % 10 == 0:
            print(f"loss = {float(loss)}")
            # the prediction of the first 10 chars always match the autoregressive prediction
            pred = net(tokens[None, :10])[0]
            print(tokens[1:11])
            print(pred.argmax(-1))
            print([int(net(tokens[None, :n+1])[0, -1].argmax())  # this is autoregressive prediction of the first 10 chars
                  for n in range(10)])
            print()
            print(f"loss = {float(loss)}")
            pred = net(tokens[None, :1])[0]
            print(tokens[1:2].cpu().detach().numpy())
            print(pred.argmax(-1).cpu().detach().numpy())
            pred = net(tokens[None, :10])[0]
            print(tokens[1:11].cpu().detach().numpy())
            print(pred.argmax(-1).cpu().detach().numpy())
    pred = net(tokens[None, :10])[0]
    print(tokens[1:11].cpu().detach().numpy())
    print(pred.argmax(-1).cpu().detach().numpy())
    torch.save(net, 'transformer.pth')


def sample():
    import sys
    net = torch.load('transformer.pth', weights_only=False)
    net.cuda()
    data = [127]
    for i in range(10000):
        tokens = torch.as_tensor(data[-500:]).cuda()
        pred = net(tokens[None])[0, -1]
        # next_char = torch.multinomial(torch.softmax(pred, -1), 1)
        next_char = pred.argmax(-1)
        if next_char == 0:
            break
        data.append(int(next_char))
        sys.stdout.write(chr(int(next_char)))
        sys.stdout.flush()


# net = Transformer(128, 8, 4)
# net(torch.rand(16, 10, 128)).shape
if __name__ == "__main__":
    train()
    sample()
    # learned_embed = torch.arange(6).float().view(2, 3)
    # print(attention_mask(5, learned_embed))
