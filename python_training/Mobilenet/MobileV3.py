import torch
import torch.nn as nn
import torch.nn.functional as F

class HSwish(nn.Module):
    def forward(self, x):
        return x * F.relu6(x + 3) / 6

class SEBlock(nn.Module):
    def __init__(self, in_channels, reduction=4):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y

class MobileNetV3Block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, expand_ratio, use_se, activation):
        super(MobileNetV3Block, self).__init__()
        self.use_residual = (stride == 1 and in_channels == out_channels)
        hidden_dim = int(round(in_channels * expand_ratio))
        self.use_se = use_se
        self.activation = activation()

        self.expand = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, 1, 1, 0, bias=False),
            nn.BatchNorm2d(hidden_dim),
            self.activation
        ) if expand_ratio != 1 else nn.Identity()

        self.depthwise = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size, stride, kernel_size // 2, groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            self.activation
        )

        self.se = SEBlock(hidden_dim) if use_se else nn.Identity()

        self.project = nn.Sequential(
            nn.Conv2d(hidden_dim, out_channels, 1, 1, 0, bias=False),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        out = self.expand(x)
        out = self.depthwise(out)
        out = self.se(out)
        out = self.project(out)
        if self.use_residual:
            return x + out
        else:
            return out

class MobileNetV3(nn.Module):
    def __init__(self, num_classes=1000, mode='small'):
        super(MobileNetV3, self).__init__()

        if mode == 'small':
            self.cfgs = [
                # k, exp, c, se, nl, s
                [3, 16, 16, True, nn.ReLU, 2],
                [3, 72, 24, False, nn.ReLU, 2],
                [3, 88, 24, False, nn.ReLU, 1],
                [5, 96, 40, True, HSwish, 2],
                [5, 240, 40, True, HSwish, 1],
                [5, 240, 40, True, HSwish, 1],
                [5, 120, 48, True, HSwish, 1],
                [5, 144, 48, True, HSwish, 1],
                [5, 288, 96, True, HSwish, 2],
                [5, 576, 96, True, HSwish, 1],
                [5, 576, 96, True, HSwish, 1],
            ]
            last_channel = 1024
        else:  # large model, 这里省略

            raise NotImplementedError("Only small mode is implemented here.")

        self.stem = nn.Sequential(
            nn.Conv2d(3, 16, 3, 2, 1, bias=False),
            nn.BatchNorm2d(16),
            HSwish()
        )

        layers = []
        input_channel = 16
        for k, exp, c, se, nl, s in self.cfgs:
            layers.append(MobileNetV3Block(input_channel, c, k, s, exp / input_channel, se, nl))
            input_channel = c
        self.blocks = nn.Sequential(*layers)

        self.conv = nn.Sequential(
            nn.Conv2d(input_channel, last_channel, 1, 1, 0, bias=False),
            nn.BatchNorm2d(last_channel),
            HSwish()
        )

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Linear(last_channel, 1280),
            HSwish(),
            nn.Dropout(0.2),
            nn.Linear(1280, num_classes),
        )

        self._initialize_weights()

    def forward(self, x):
        x = self.stem(x)
        x = self.blocks(x)
        x = self.conv(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

if __name__ == '__main__':
    model = MobileNetV3(num_classes=10, mode='small')
    print(model)
    input_tensor = torch.randn(1, 3, 224, 224)
    output = model(input_tensor)
    print(output.shape)
