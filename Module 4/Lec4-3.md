```python
import torch
```


```python
class MyModel(torch.nn.Module):
    def __init__(self, layer_size=[512, 512, 512]) -> None:
        super().__init__()
        layers = []
        layers.append(torch.nn.Flatten())
        c = 128*128*3  # number of channel
        for s in layer_size:
            layers.append(torch.nn.Linear(c, s))
            layers.append(torch.nn.ReLU())
            c = s
        layers.append(torch.nn.Linear(c, 102))

        self.model = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

net = MyModel()
x = torch.randn(10, 3, 128, 128)
net(x)
```




    tensor([[ 0.0540,  0.0027,  0.0965,  ..., -0.0208, -0.0392,  0.0374],
            [ 0.0329, -0.0074,  0.0568,  ..., -0.0303, -0.0554,  0.0392],
            [ 0.0187, -0.0077,  0.0778,  ...,  0.0313,  0.0067,  0.0371],
            ...,
            [-0.0097, -0.0534,  0.0567,  ..., -0.0285, -0.0168, -0.0027],
            [ 0.0098, -0.0306,  0.0328,  ..., -0.0407, -0.0280, -0.0087],
            [ 0.0314, -0.0398,  0.0219,  ..., -0.0092,  0.0210,  0.0214]],
           grad_fn=<AddmmBackward0>)




```python
class MyModelNoBias(torch.nn.Module):
    def __init__(self, layer_size=[512, 512, 512]) -> None:
        super().__init__()
        layers = []
        layers.append(torch.nn.Flatten())
        c = 128*128*3  # number of channel
        for s in layer_size:
            layers.append(torch.nn.Linear(c, s, bias=False))
            layers.append(torch.nn.ReLU())
            c = s
        layers.append(torch.nn.Linear(c, 102, bias=False))

        self.model = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


net = MyModel()
x = torch.randn(10, 3, 128, 128)
net(x)
```




    tensor([[ 0.0501, -0.0281,  0.0328,  ...,  0.0765,  0.0261,  0.0109],
            [ 0.0390, -0.0766,  0.0622,  ...,  0.0773, -0.0082,  0.0123],
            [ 0.0679, -0.0566, -0.0129,  ...,  0.0461,  0.0135, -0.0020],
            ...,
            [ 0.0884, -0.0320,  0.0412,  ...,  0.0822, -0.0115,  0.0615],
            [ 0.0412, -0.0194,  0.0123,  ...,  0.0929, -0.0107,  0.0389],
            [ 0.0588, -0.0306,  0.0452,  ...,  0.0779, -0.0029,  0.0372]],
           grad_fn=<AddmmBackward0>)




```python
net0 = MyModel([])
x = torch.randn(10, 3, 128, 128)
print(f"{net0(x).norm()=}")
net1 = MyModel([512])
print(f"{net1(x).norm()=}")
net2 = MyModel([512, 512])
print(f"{net2(x).norm()=}")

for n in range(10):
    netn = MyModel([512] * n)
    print(f"{n} {netn(x).norm()=}")
```

    net0(x).norm()=tensor(18.3989, grad_fn=<LinalgVectorNormBackward0>)
    net1(x).norm()=tensor(7.2410, grad_fn=<LinalgVectorNormBackward0>)
    net2(x).norm()=tensor(3.1070, grad_fn=<LinalgVectorNormBackward0>)
    0 netn(x).norm()=tensor(18.6123, grad_fn=<LinalgVectorNormBackward0>)
    1 netn(x).norm()=tensor(7.2068, grad_fn=<LinalgVectorNormBackward0>)
    2 netn(x).norm()=tensor(3.4652, grad_fn=<LinalgVectorNormBackward0>)
    3 netn(x).norm()=tensor(1.5229, grad_fn=<LinalgVectorNormBackward0>)
    4 netn(x).norm()=tensor(0.9647, grad_fn=<LinalgVectorNormBackward0>)
    5 netn(x).norm()=tensor(0.8190, grad_fn=<LinalgVectorNormBackward0>)
    6 netn(x).norm()=tensor(0.9076, grad_fn=<LinalgVectorNormBackward0>)
    7 netn(x).norm()=tensor(0.8148, grad_fn=<LinalgVectorNormBackward0>)
    8 netn(x).norm()=tensor(0.9776, grad_fn=<LinalgVectorNormBackward0>)
    9 netn(x).norm()=tensor(0.8660, grad_fn=<LinalgVectorNormBackward0>)



```python
x = torch.randn(10, 3, 128, 128)

for n in range(10):
    netn = MyModelNoBias([512] * n)
    print(f"{n} {netn(x).norm()=}")
```

    0 netn(x).norm()=tensor(18.0155, grad_fn=<LinalgVectorNormBackward0>)
    1 netn(x).norm()=tensor(7.1321, grad_fn=<LinalgVectorNormBackward0>)
    2 netn(x).norm()=tensor(3.0214, grad_fn=<LinalgVectorNormBackward0>)
    3 netn(x).norm()=tensor(1.2086, grad_fn=<LinalgVectorNormBackward0>)
    4 netn(x).norm()=tensor(0.5479, grad_fn=<LinalgVectorNormBackward0>)
    5 netn(x).norm()=tensor(0.2147, grad_fn=<LinalgVectorNormBackward0>)
    6 netn(x).norm()=tensor(0.0866, grad_fn=<LinalgVectorNormBackward0>)
    7 netn(x).norm()=tensor(0.0344, grad_fn=<LinalgVectorNormBackward0>)
    8 netn(x).norm()=tensor(0.0118, grad_fn=<LinalgVectorNormBackward0>)
    9 netn(x).norm()=tensor(0.0062, grad_fn=<LinalgVectorNormBackward0>)



```python
class MyModelBN(torch.nn.Module):
    def __init__(self, layer_size=[512, 512, 512]) -> None:
        super().__init__()
        layers = []
        layers.append(torch.nn.Flatten())
        c = 128*128*3  # number of channel
        for s in layer_size:
            layers.append(torch.nn.Linear(c, s, bias=False))
            layers.append(torch.nn.BatchNorm1d(s)) #batch normalization
            layers.append(torch.nn.ReLU())
            c = s
        layers.append(torch.nn.Linear(c, 102, bias=False))

        self.model = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


x = torch.randn(10, 3, 128, 128)

for n in range(10):
    netn = MyModelBN([512] * n)
    print(f"{n} {netn(x).norm()=}")
```

    0 netn(x).norm()=tensor(18.5365, grad_fn=<LinalgVectorNormBackward0>)
    1 netn(x).norm()=tensor(12.9939, grad_fn=<LinalgVectorNormBackward0>)
    2 netn(x).norm()=tensor(13.2019, grad_fn=<LinalgVectorNormBackward0>)
    3 netn(x).norm()=tensor(12.9503, grad_fn=<LinalgVectorNormBackward0>)
    4 netn(x).norm()=tensor(13.4037, grad_fn=<LinalgVectorNormBackward0>)
    5 netn(x).norm()=tensor(12.8535, grad_fn=<LinalgVectorNormBackward0>)
    6 netn(x).norm()=tensor(12.9624, grad_fn=<LinalgVectorNormBackward0>)
    7 netn(x).norm()=tensor(13.3393, grad_fn=<LinalgVectorNormBackward0>)
    8 netn(x).norm()=tensor(13.2729, grad_fn=<LinalgVectorNormBackward0>)
    9 netn(x).norm()=tensor(13.0163, grad_fn=<LinalgVectorNormBackward0>)



```python
class MyModelLN(torch.nn.Module):
    def __init__(self, layer_size=[512, 512, 512]) -> None:
        super().__init__()
        layers = []
        layers.append(torch.nn.Flatten())
        c = 128*128*3  # number of channel
        for s in layer_size:
            layers.append(torch.nn.LayerNorm(s, bias=False))  # can put the normalization here

            layers.append(torch.nn.Linear(c, s))
            # layers.append(torch.nn.LayerNorm(s))  # can put the normalization here
            layers.append(torch.nn.ReLU())
            c = s
        layers.append(torch.nn.Linear(c, 102, bias=False))

        self.model = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)
    

for n in range(30):
    netn = MyModelLN([512] * n)
    print(f"{n} {netn(x).norm()=}")
```

    0 netn(x).norm()=tensor(18.0582, grad_fn=<LinalgVectorNormBackward0>)
    1 netn(x).norm()=tensor(12.7572, grad_fn=<LinalgVectorNormBackward0>)
    2 netn(x).norm()=tensor(12.5842, grad_fn=<LinalgVectorNormBackward0>)
    3 netn(x).norm()=tensor(13.0634, grad_fn=<LinalgVectorNormBackward0>)
    4 netn(x).norm()=tensor(12.5875, grad_fn=<LinalgVectorNormBackward0>)
    5 netn(x).norm()=tensor(12.1720, grad_fn=<LinalgVectorNormBackward0>)
    6 netn(x).norm()=tensor(14.2014, grad_fn=<LinalgVectorNormBackward0>)
    7 netn(x).norm()=tensor(12.3908, grad_fn=<LinalgVectorNormBackward0>)
    8 netn(x).norm()=tensor(14.5423, grad_fn=<LinalgVectorNormBackward0>)
    9 netn(x).norm()=tensor(12.3325, grad_fn=<LinalgVectorNormBackward0>)
    10 netn(x).norm()=tensor(13.5382, grad_fn=<LinalgVectorNormBackward0>)
    11 netn(x).norm()=tensor(13.0165, grad_fn=<LinalgVectorNormBackward0>)
    12 netn(x).norm()=tensor(13.2984, grad_fn=<LinalgVectorNormBackward0>)
    13 netn(x).norm()=tensor(13.6450, grad_fn=<LinalgVectorNormBackward0>)
    14 netn(x).norm()=tensor(12.3181, grad_fn=<LinalgVectorNormBackward0>)
    15 netn(x).norm()=tensor(13.2800, grad_fn=<LinalgVectorNormBackward0>)
    16 netn(x).norm()=tensor(12.2531, grad_fn=<LinalgVectorNormBackward0>)
    17 netn(x).norm()=tensor(11.7266, grad_fn=<LinalgVectorNormBackward0>)
    18 netn(x).norm()=tensor(12.2116, grad_fn=<LinalgVectorNormBackward0>)
    19 netn(x).norm()=tensor(12.3550, grad_fn=<LinalgVectorNormBackward0>)
    20 netn(x).norm()=tensor(13.2683, grad_fn=<LinalgVectorNormBackward0>)
    21 netn(x).norm()=tensor(13.2982, grad_fn=<LinalgVectorNormBackward0>)
    22 netn(x).norm()=tensor(11.5966, grad_fn=<LinalgVectorNormBackward0>)
    23 netn(x).norm()=tensor(13.7284, grad_fn=<LinalgVectorNormBackward0>)
    24 netn(x).norm()=tensor(13.0782, grad_fn=<LinalgVectorNormBackward0>)
    25 netn(x).norm()=tensor(11.0853, grad_fn=<LinalgVectorNormBackward0>)
    26 netn(x).norm()=tensor(12.7664, grad_fn=<LinalgVectorNormBackward0>)
    27 netn(x).norm()=tensor(11.3610, grad_fn=<LinalgVectorNormBackward0>)
    28 netn(x).norm()=tensor(14.9791, grad_fn=<LinalgVectorNormBackward0>)
    29 netn(x).norm()=tensor(12.8191, grad_fn=<LinalgVectorNormBackward0>)

