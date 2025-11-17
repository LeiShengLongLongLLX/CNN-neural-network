import torch
import torch.nn as nn
import torch.nn.functional as F

class DepthwiseSeparableConv(nn.Module):
    """深度可分离卷积：先深度卷积，再逐点卷积"""
    def __init__(self, in_channels, out_channels, stride=1):
        super(DepthwiseSeparableConv, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=3, 
                                   stride=stride, padding=1, groups=in_channels, bias=False)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, 
                                   stride=1, padding=0, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        
    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        return F.relu(x)

class MobileNetV1(nn.Module):
    def __init__(self, num_classes=1000):
        super(MobileNetV1, self).__init__()
        # 输入是3通道RGB图片，大小一般是224x224
        self.model = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False), # 32x32
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            
            DepthwiseSeparableConv(32, 64, stride=1),   # 112x112
            DepthwiseSeparableConv(64, 128, stride=2),  # 56x56
            DepthwiseSeparableConv(128, 128, stride=1), # 56x56
            DepthwiseSeparableConv(128, 256, stride=2), # 28x28
            DepthwiseSeparableConv(256, 256, stride=1), # 28x28
            DepthwiseSeparableConv(256, 512, stride=2), # 14x14
            
            # 连续5个相同的卷积块，输出通道512，尺寸14x14
            DepthwiseSeparableConv(512, 512, stride=1),
            DepthwiseSeparableConv(512, 512, stride=1),
            DepthwiseSeparableConv(512, 512, stride=1),
            DepthwiseSeparableConv(512, 512, stride=1),
            DepthwiseSeparableConv(512, 512, stride=1),
            
            DepthwiseSeparableConv(512, 1024, stride=2), # 7x7
            DepthwiseSeparableConv(1024, 1024, stride=1), # 7x7
            
            nn.AdaptiveAvgPool2d(1),  # 全局平均池化，输出大小1x1
        )
        self.classifier = nn.Linear(1024, num_classes)
    
    def forward(self, x):
        x = self.model(x)
        x = x.view(x.size(0), -1)  # flatten成(batch_size, 1024)
        x = self.classifier(x)
        return x

if __name__ == '__main__':
    net = MobileNetV1(num_classes=10)  # 比如10分类
    print(net)
    
    # 测试网络输入输出尺寸
    x = torch.randn(1, 3, 224, 224)  # batch=1, 3通道，224x224
    y = net(x)
    print(y.shape)  # 输出 torch.Size([1, 10])
