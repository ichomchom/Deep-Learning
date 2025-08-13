```python
import torch
```


```python
class MyModelLN(torch.nn.Module):
    class Block(torch.nn.Module):
        def __init__(self, in_channels, out_channels) -> None:
            super().__init__()
            self.linear = torch.nn.Linear(in_channels, out_channels) ## convolution
            self.norm = torch.nn.LayerNorm(out_channels) ## normalization
            self.relu = torch.nn.ReLU() ## ReLU
            if in_channels != out_channels:
                self.skip = torch.nn.Linear(in_channels, out_channels)
            else:
                self.skip = torch.nn.Identity()

        def forward(self, x):
            y = self.relu(self.norm(self.linear(x)))
            return self.skip(x) + y ## add skip connection
            ## add x at the end changes this model from layer normalize network to residual network

    def __init__(self, layer_size=[512, 512, 512]) -> None:
        super().__init__()
        layers = []
        layers.append(torch.nn.Flatten())
        c = 128*128*3  # number of channel
        layers.append(torch.nn.Linear(c, 512, bias=False))
        c = 512

        for s in layer_size:
            layers.append(self.Block(c, s))
            c = s
        layers.append(torch.nn.Linear(c, 102, bias=False))

        self.model = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


## Residual networks have skip connections in them that skip a bunch of sequential layers
## The easiest way to implement is group linear layer norm and ReLU together in a layer
# Then add skip connections around
x = torch.randn(10, 3, 128, 128)
net = MyModelLN([512] * 4)
print(net(x))
# for n in range(30):
#     netn = MyModelLN([512] * n)
#     print(f"{n} {netn(x).norm()=}")
```

    tensor([[ 8.4396e-01, -2.1168e+00, -1.3959e-01,  ...,  4.6358e-01,
             -9.7295e-02,  2.2572e+00],
            [ 1.8924e+00, -8.4603e-01,  5.5874e-01,  ...,  2.8706e-01,
             -1.0950e+00,  2.2275e+00],
            [ 5.7053e-01, -1.5183e+00, -2.0056e-01,  ...,  5.9660e-01,
             -6.0130e-01,  1.7595e+00],
            ...,
            [ 5.6692e-01, -6.7182e-01,  1.3412e+00,  ...,  8.4679e-01,
              1.7534e-01,  1.9191e+00],
            [ 1.6121e-01, -9.9722e-01,  4.8937e-01,  ...,  5.4266e-01,
             -2.1506e-01,  1.3463e+00],
            [ 1.8894e-03, -1.9166e+00,  2.2276e-01,  ..., -6.3081e-02,
             -8.6957e-01, -8.4825e-01]], grad_fn=<MmBackward0>)



```python
class MyModelLN(torch.nn.Module):
    class Block(torch.nn.Module):
        def __init__(self, in_channels, out_channels) -> None:
            super().__init__()
            self.linear1 = torch.nn.Linear(
                in_channels, out_channels)  # convolution
            self.norm1 = torch.nn.LayerNorm(out_channels)  # normalization
            self.relu1 = torch.nn.ReLU()  # ReLU
            self.linear2 = torch.nn.Linear(
                in_channels, out_channels)  # convolution
            self.norm2 = torch.nn.LayerNorm(out_channels)  # normalization
            self.relu2 = torch.nn.ReLU()  # ReLU
            if in_channels != out_channels:
                self.skip = torch.nn.Linear(in_channels, out_channels)
            else:
                self.skip = torch.nn.Identity()

        def forward(self, x):
            y = self.relu1(self.norm1(self.linear1(x)))
            y = self.relu2(self.norm2(self.linear2(x)))
            return self.skip(x) + y  # add skip connection
            # add x at the end changes this model from layer normalize network to residual network

    def __init__(self, layer_size=[512, 512, 512]) -> None:
        super().__init__()
        layers = []
        layers.append(torch.nn.Flatten())
        c = 128*128*3  # number of channel
        layers.append(torch.nn.Linear(c, 512, bias=False))
        c = 512

        for s in layer_size:
            layers.append(self.Block(c, s))
            c = s
        layers.append(torch.nn.Linear(c, 102, bias=False))

        self.model = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


# Residual networks have skip connections in them that skip a bunch of sequential layers
# The easiest way to implement is group linear layer norm and ReLU together in a layer
# Then add skip connections around
x = torch.randn(10, 3, 128, 128)
net = MyModelLN([512] * 4)
print(net(x))
# for n in range(30):
#     netn = MyModelLN([512] * n)
#     print(f"{n} {netn(x).norm()=}")
```

    tensor([[-0.2662,  0.2716, -0.0291,  ...,  0.3620,  1.0407, -1.2572],
            [ 0.6800,  0.7348, -0.0885,  ...,  0.5710,  1.4504, -1.5437],
            [-0.0852, -0.6613, -0.1593,  ...,  1.0970,  1.3200, -0.8434],
            ...,
            [-0.6748,  0.5078, -0.2419,  ...,  0.5855,  1.5774, -2.5527],
            [-0.9504, -0.4432, -0.0864,  ...,  0.7035,  2.7226, -1.7251],
            [-0.2463,  1.6944, -0.3771,  ...,  1.8689,  1.4131, -1.5296]],
           grad_fn=<MmBackward0>)



```python

```


```python
class MyModelLN(torch.nn.Module):
    class Block(torch.nn.Module):
        def __init__(self, in_channels, out_channels) -> None:
            super().__init__()
            self.model = torch.nn.Sequential(torch.nn.Linear(in_channels, out_channels), 
                                             torch.nn.LayerNorm(out_channels), torch.nn.ReLU(), 
                                             torch.nn.Linear(in_channels, out_channels), 
                                             torch.nn.LayerNorm(out_channels), torch.nn.ReLU())
            if in_channels != out_channels:
                self.skip = torch.nn.Linear(in_channels, out_channels)
            else:
                self.skip = torch.nn.Identity()

        def forward(self, x):
            y = self.relu1(self.norm1(self.linear1(x)))
            y = self.relu2(self.norm2(self.linear2(x)))
            return self.skip(x) + y  # add skip connection
            # add x at the end changes this model from layer normalize network to residual network

    def __init__(self, layer_size=[512, 512, 512]) -> None:
        super().__init__()
        layers = []
        layers.append(torch.nn.Flatten())
        c = 128*128*3  # number of channel
        layers.append(torch.nn.Linear(c, 512, bias=False))
        c = 512

        for s in layer_size:
            layers.append(self.Block(c, s))
            c = s
        layers.append(torch.nn.Linear(c, 102, bias=False))

        self.model = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


# Residual networks have skip connections in them that skip a bunch of sequential layers
# The easiest way to implement is group linear layer norm and ReLU together in a layer
# Then add skip connections around
x = torch.randn(10, 3, 128, 128)
net = MyModelLN([512] * 4)
print(net(x))
# for n in range(30):
#     netn = MyModelLN([512] * n)
#     print(f"{n} {netn(x).norm()=}")
```
