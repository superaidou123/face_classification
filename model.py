import paddle
from paddle import nn
from paddle.nn import functional as F


def params(l2_regularization):
    _params = paddle.ParamAttr(regularizer=paddle.regularizer.L2Decay(l2_regularization))
    return _params


class SimpleCNN(nn.Layer):
    def __init__(self, in_channels, num_classes):
        super(SimpleCNN, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2D(in_channels, 16, 7, 1, 3),
            nn.BatchNorm2D(16),
            nn.Conv2D(16, 16, 7, 1, 3),
            nn.BatchNorm2D(16),
            nn.ReLU(),
            nn.AvgPool2D(2),
            nn.Dropout(0.5),

            nn.Conv2D(16, 32, 5, 1, 2),
            nn.BatchNorm2D(32),
            nn.Conv2D(32, 32, 5, 1, 2),
            nn.BatchNorm2D(32),
            nn.ReLU(),
            nn.AvgPool2D(2),
            nn.Dropout(0.5),

            nn.Conv2D(32, 64, 3, 1, 1),
            nn.BatchNorm2D(64),
            nn.Conv2D(64, 64, 3, 1, 1),
            nn.BatchNorm2D(64),
            nn.ReLU(),
            nn.AvgPool2D(2),
            nn.Dropout(0.5),

            nn.Conv2D(64, 128, 3, 1, 1),
            nn.BatchNorm2D(128),
            nn.Conv2D(128, 128, 3, 1, 1),
            nn.BatchNorm2D(128),
            nn.ReLU(),
            nn.AvgPool2D(2),
            nn.Dropout(0.5),

            nn.Conv2D(128, 256, 3, 1, 1),
            nn.BatchNorm2D(256),
            nn.Conv2D(256, num_classes, 3, 1, 1),
            nn.AdaptiveAvgPool2D(1)
        )

    def forward(self, x):
        x = self.model(x)
        x = x.squeeze(axis=-1).squeeze(axis=-1)
        x = F.softmax(x)
        return x


class SimplerCNN(nn.Layer):
    def __init__(self, in_channels, num_classes):
        super(SimplerCNN, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2D(in_channels, 16, 5, 1, 2),
            nn.BatchNorm2D(16),
            nn.Conv2D(16, 16, 5, 2, 2),
            nn.BatchNorm2D(16),
            nn.ReLU(),
            nn.Dropout(0.25),

            nn.Conv2D(16, 32, 5, 1, 2),
            nn.BatchNorm2D(32),
            nn.Conv2D(32, 32, 5, 2, 2),
            nn.BatchNorm2D(32),
            nn.ReLU(),
            nn.Dropout(0.25),

            nn.Conv2D(32, 64, 3, 1, 1),
            nn.BatchNorm2D(64),
            nn.Conv2D(64, 64, 3, 2, 1),
            nn.BatchNorm2D(64),
            nn.ReLU(),
            nn.Dropout(0.25),

            nn.Conv2D(64, 64, 1, 1),
            nn.BatchNorm2D(64),
            nn.Conv2D(64, 128, 3, 2, 1),
            nn.BatchNorm2D(128),
            nn.ReLU(),
            nn.Dropout(0.25),

            nn.Conv2D(128, 256, 1, 1),
            nn.BatchNorm2D(256),
            nn.Conv2D(256, 128, 3, 2, 1),

            nn.Conv2D(128, 256, 1, 1),
            nn.BatchNorm2D(256),
            nn.Conv2D(256, num_classes, 3, 2, 1),
        )

    def forward(self, x):
        x = self.model(x)
        x = x.squeeze(axis=-1).squeeze(axis=-1)
        x = F.softmax(x)
        return x


class SeparableConv2D(nn.Layer):
    def __init__(self, in_channels, out_channels, kernel_size, stride, bias=False):
        super(SeparableConv2D, self).__init__()
        self.dw_conv = nn.Conv2D(in_channels, in_channels, kernel_size, stride, padding=kernel_size // 2,
                                 groups=in_channels, bias_attr=bias)
        self.point_conv = nn.Conv2D(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias_attr=bias)

    def forward(self, x):
        x = self.dw_conv(x)
        x = self.point_conv(x)
        return x


class Block(nn.Layer):
    def __init__(self, in_channels, out_channels):
        super(Block, self).__init__()
        self.residual = nn.Sequential(
            nn.Conv2D(in_channels, out_channels, 1, 2, bias_attr=False),
            nn.BatchNorm2D(out_channels)
        )

        self.SeparableConv2D1 = nn.Sequential(
            SeparableConv2D(in_channels, out_channels, 3, 1),
            nn.BatchNorm2D(out_channels),
            nn.ReLU()
        )
        self.SeparableConv2D2 = nn.Sequential(
            SeparableConv2D(out_channels, out_channels, 3, 1),
            nn.BatchNorm2D(out_channels),
        )
        self.max_pool = nn.MaxPool2D(3, 2, 1)

    def forward(self, x):
        residual = self.residual(x)
        x = self.SeparableConv2D1(x)
        x = self.SeparableConv2D2(x)
        x = self.max_pool(x)
        return residual + x


class MiniXception(nn.Layer):
    def __init__(self, in_channels, num_classes, l2_regularization=0.010):
        super(MiniXception, self).__init__()
        self.base = nn.Sequential(
            nn.Conv2D(in_channels, 8, 3, 1, weight_attr=params(l2_regularization), bias_attr=False),
            nn.BatchNorm2D(8),
            nn.ReLU(),
            nn.Conv2D(8, 8, 3, 1, weight_attr=params(l2_regularization), bias_attr=False),
            nn.BatchNorm2D(8),
            nn.ReLU()
        )
        self.block1 = Block(8, 16)
        self.block2 = Block(16, 32)
        self.block3 = Block(32, 64)
        self.block4 = Block(64, 128)
        self.conv = nn.Sequential(
            nn.Conv2D(128, num_classes, 3, 1, 1),
            nn.AdaptiveAvgPool2D(1)
        )

    def forward(self, x):
        x = self.base(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.conv(x)
        x = x.squeeze(axis=-1).squeeze(axis=-1)
        x = F.softmax(x)
        return x