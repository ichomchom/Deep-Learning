```python
import torch
```


```python
net = torch.nn.Conv2d(1, 1, 3, 2, 1)

x = torch.zeros(1, 1, 5, 5)
print(x.shape, net(x).shape)


```

    torch.Size([1, 1, 5, 5]) torch.Size([1, 1, 3, 3])



```python
net = torch.nn.Conv2d(1, 1, 3, stride=1, padding=2, dilation=2)

x = torch.zeros(1, 1, 5, 5)
print(x.shape, net(x).shape)
```

    torch.Size([1, 1, 5, 5]) torch.Size([1, 1, 5, 5])



```python
net1 = torch.nn.Sequential(
    torch.nn.Conv2d(1, 1, 3, stride=2, padding=1, dilation=1),
    torch.nn.ReLU(),
    torch.nn.Conv2d(1, 1, 3, stride=1, padding=1, dilation=1)
)
net2 = torch.nn.Sequential(
    torch.nn.Conv2d(1, 1, 3, stride=1, padding=1, dilation=1),
    torch.nn.ReLU(),
    torch.nn.Conv2d(1, 1, 3, stride=1, padding=2, dilation=2)
)
# dilation net2
# only dilate the 2nd kernel, the first kernel needs to be the same kernel
# in net 1, we strided, so we skipped every other output
# now in net 2, when we use dilation, we now need to dilate the kernel that comes after
# when we dilate, we need to increase the padding
for p1, p2 in zip(net1.parameters(), net2.parameters()):
    p2.data = p1.data.clone()
    p2.requires_grad = True

print(list(net1.parameters()))
print(list(net2.parameters()))

x = torch.randn(1, 1, 5, 5)
print(x.shape, net1(x).shape)
print(x.shape, net2(x).shape)
y1  = net1(x)
y2 = net2(x)
print(y1)
print(y2[:, :, ::2, ::2])
```

    [Parameter containing:
    tensor([[[[ 0.2249,  0.1837, -0.1058],
              [-0.2648, -0.0549,  0.1453],
              [ 0.2576,  0.2293,  0.2288]]]], requires_grad=True), Parameter containing:
    tensor([0.2085], requires_grad=True), Parameter containing:
    tensor([[[[ 0.0833,  0.2698, -0.2645],
              [ 0.0073,  0.1843,  0.0281],
              [ 0.1993, -0.2963, -0.0175]]]], requires_grad=True), Parameter containing:
    tensor([-0.1083], requires_grad=True)]
    [Parameter containing:
    tensor([[[[ 0.2249,  0.1837, -0.1058],
              [-0.2648, -0.0549,  0.1453],
              [ 0.2576,  0.2293,  0.2288]]]], requires_grad=True), Parameter containing:
    tensor([0.2085], requires_grad=True), Parameter containing:
    tensor([[[[ 0.0833,  0.2698, -0.2645],
              [ 0.0073,  0.1843,  0.0281],
              [ 0.1993, -0.2963, -0.0175]]]], requires_grad=True), Parameter containing:
    tensor([-0.1083], requires_grad=True)]
    torch.Size([1, 1, 5, 5]) torch.Size([1, 1, 3, 3])
    torch.Size([1, 1, 5, 5]) torch.Size([1, 1, 5, 5])
    tensor([[[[-0.0923, -0.0877, -0.0834],
              [-0.3334, -0.0011,  0.0278],
              [-0.0735,  0.0134, -0.0729]]]], grad_fn=<ConvolutionBackward0>)
    tensor([[[[-0.0923, -0.0877, -0.0834],
              [-0.3334, -0.0011,  0.0278],
              [-0.0735,  0.0134, -0.0729]]]], grad_fn=<SliceBackward0>)



```python
net1 = torch.nn.Sequential(
    torch.nn.Conv2d(1, 1, 3, stride=2, padding=1, dilation=1),
    torch.nn.ReLU(),
    torch.nn.Conv2d(1, 1, 3, stride=1, padding=1, dilation=1),
    torch.nn.ReLU(),
    torch.nn.ConvTranspose2d(1, 1, 3, stride=2, padding=1, output_padding=1) # stride here means, how much you want to upsample, 
                                                                        # padding here is cut and shrink the output. padding 0 will make output larger
)
x = torch.randn(1, 1, 5, 5)
print(net1(x)[:, :, :x.shape[2], :x.shape[3]].shape)
```

    torch.Size([1, 1, 5, 5])

