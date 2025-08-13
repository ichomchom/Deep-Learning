```python
import torch
```


```python
class ConvNet(torch.nn.Module):
    class Block(torch.nn.Module):
        def __init__(self, in_channels, out_channels, stride) -> None:
            super().__init__()
            kernel_size = 3
            padding = (kernel_size - 1)//2

            # all convolutional layers only first layer take input channels, the rest take out channels
            # only the first one stride the other layers will not stride
            self.c1 = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
            self.c2 = torch.nn.Conv2d(out_channels, out_channels, kernel_size, 1, padding)
            self.c3 = torch.nn.Conv2d(out_channels, out_channels, kernel_size, 1, padding)
            self.relu = torch.nn.ReLU()
            # we can start adding normalizations in here
            # can add residual connections
        def forward(self, x):
            x = self.relu(self.c1(x))
            x = self.relu(self.c2(x))
            x = self.relu(self.c3(x))
            return x

    def __init__(self, channels_l0=64, n_blocks=4) -> None:  # channels_l0 channel size of the first layer
        super().__init__()
        cnn_layers = [
            torch.nn.Conv2d(3, channels_l0, kernel_size=11, stride=2, padding=5), # special layer, input is channel of 1st layer, stride factor of 2
            torch.nn.ReLU(),
        ]
        c1 = channels_l0 #input channel
        for _ in range(n_blocks):
            c2 = c1 * 2 # increase the channel by factor of 2
            cnn_layers.append(self.Block(c1, c2, stride=2)) # stride by factor of 2
            c1 = c2
        cnn_layers.append(torch.nn.Conv2d(c1, 1, kernel_size=1))
        # cnn_layers.append(torch.nn.AdaptiveAvgPool2d(1))

        self.network = torch.nn.Sequential(*cnn_layers)

    def forward(self, x):
        return self.network(x)


net = ConvNet(n_blocks=3)
x = torch.randn(1, 3, 64, 64)
net(x).shape
print(net)
## Baisc structure of the Convnet will almost always be
## 1 convolution as input, non-linearity
## then you have bunch of blocks that do the heavy lifting inside
## then 1 convolutional layer as output
## if you interested in getting 1 single classification output
# you will add AdaptiveAvgPool2d which is global average pooling

```

    ConvNet(
      (network): Sequential(
        (0): Conv2d(3, 64, kernel_size=(11, 11), stride=(2, 2), padding=(5, 5))
        (1): ReLU()
        (2): Block(
          (c1): Conv2d(64, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
          (c2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (c3): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (relu): ReLU()
        )
        (3): Block(
          (c1): Conv2d(128, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
          (c2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (c3): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (relu): ReLU()
        )
        (4): Block(
          (c1): Conv2d(256, 512, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
          (c2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (c3): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (relu): ReLU()
        )
        (5): Conv2d(512, 1, kernel_size=(1, 1), stride=(1, 1))
      )
    )



```python

```
