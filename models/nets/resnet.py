import torch
from torch import nn
from torch.nn import BatchNorm2d
from torch.nn import functional as F

class ResNet(nn.Module):
    def __init__(self, 
        depth=50,
        out_features=['res2', 'res3', 'res4', 'res5'],
        norm="BN"):
        """
        Build layers of ResNet
        """
        super(ResNet, self).__init__()
        assert depth in (18, 34, 50, 101)

        self.init_stride = 2
        self.init_channel = 3
        self._num_layers = {
            18: (2, 2, 2, 2),
            34: (3, 4, 6, 3),
            50: (3, 4, 6, 3),
            101: (3, 4, 23, 3)
        }[depth]
        self._in_channels = (64, 256, 512, 1024)
        self._out_channels = (256, 512, 1024, 2048)
        self.out_features = out_features

        self.stem = StemBlock
        self.block = BasicBlock if depth in (18, 34) else BottleneckBlock

        self.C1 = self.stem(
            in_channels=self.init_channel,
            out_channels=self._in_channels[0]
        )

        self.C2 = self.make_stage(
            self.block,
            self._num_layers[0], 
            self._in_channels[0],
            self._out_channels[0],
            stride=1)
        
        self.C3 = self.make_stage(
            self.block,
            self._num_layers[1], 
            self._in_channels[1],
            self._out_channels[1],
            stride=2)
        
        self.C4 = self.make_stage(
            self.block,
            self._num_layers[2], 
            self._in_channels[2],
            self._out_channels[2],
            stride=2)

        self.C5 = self.make_stage(
            self.block,
            self._num_layers[3], 
            self._in_channels[3],
            self._out_channels[3],
            stride=2)

    def forward(self, x):
        output_stages = {}

        # stem
        x = self.C1(x)
        output_stages["stem"] = x

        # res2
        x = self.C2(x)
        output_stages["res2"] = x

        # res3
        x = self.C3(x)
        output_stages["res3"] = x

        # res4
        x = self.C4(x)
        output_stages["res4"] = x

        # res5
        x = self.C5(x)
        output_stages["res5"] = x

        return {key: output_stages[key] for key in self.out_features}

    def make_stage(self, block, layers, in_channels, out_channels, stride):
        strides = [stride if (n == 0) else 1 for n in range(layers)]
        layers = []
        for _stride in strides:
            layers.append(block(in_channels, out_channels, _stride))
            in_channels = out_channels
        
        return nn.Sequential(*layers)

class BottleneckBlock(nn.Module):
    # for deeper networks than 50 layers
    sampling_rate = 4
    def __init__(self, in_channels, out_channels, stride):
        super(BottleneckBlock, self).__init__()

        self._bottleneck_channels = int(out_channels / self.sampling_rate)

        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=stride,
                    bias=False
                ), 
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.shortcut = None

        self.relu = nn.ReLU(inplace=True)
        
        self.conv1 = nn.Conv2d(
            in_channels,
            self._bottleneck_channels,
            kernel_size=1,
            stride=stride
        )

        self.bn1 = nn.BatchNorm2d(
            self._bottleneck_channels,
            eps = 1e-05
        )

        self.conv2 = nn.Conv2d(
            self._bottleneck_channels,
            self._bottleneck_channels,
            kernel_size=3,
            stride=1,
            padding=1
        )

        self.bn2 = nn.BatchNorm2d(
            self._bottleneck_channels,
            eps=1e-05
        )

        self.conv3 = nn.Conv2d(
            self._bottleneck_channels,
            out_channels,
            kernel_size=1,
            stride=1
        )

        self.bn3 = nn.BatchNorm2d(
            out_channels,
            eps=1e-05
        )

    def forward(self, x):
        out = x

        out = self.conv1(out)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.shortcut is not None:
            res = self.shortcut(x)
            out = out + res
        else:
            out = out + x

        out = self.relu(out)

        return out

class BasicBlock(nn.Module):
    """
    Basic residual block for ResNet18 and ResNet34.
    Two 3*3 conv, projection shortcut if needed.
    """
    def __init__(self):
        pass

class StemBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(StemBlock, self).__init__()
        self.C1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(out_channels, eps=0.00001, momentum=0.01),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.C1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, kernel_size=3, stride=2, padding=1)
        return x

def test():
    net = ResNet(depth=50)
    # print(net)
    y = net(torch.randn(1, 3, 800, 800))
    
    for key in y.keys():
        print(key, y[key].shape)

if __name__ == "__main__":
    test()
