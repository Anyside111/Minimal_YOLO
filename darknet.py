import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import MaxPool2d


class GlobalAvgPool2d(torch.nn.Module):
    def __init__(self):
        super(GlobalAvgPool2d, self).__init__()

    @staticmethod
    def forward(x):
        return torch.mean(x.view(x.size(0), x.size(1), -1), dim=2)


def auto_pad(k, p=None):
    # Pad to 'same'
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


class Conv(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):
        super(Conv, self).__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, auto_pad(k, p), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.LeakyReLU(0.01) if act else nn.ReLU()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class DarkNet(nn.Module):
    def __init__(self, anchor_size=10):
        super().__init__()

        self.stage_1 = nn.Sequential(Conv(3, 32, 3), MaxPool2d(2, 2), Conv(32, 32, 3), MaxPool2d(2, 2))
        self.stage_2 = nn.Sequential(Conv(32, 64, 3), MaxPool2d(2, 2), Conv(64, 64, 3), MaxPool2d(2, 2))

        self.stage_3 = nn.Sequential(Conv(64, 128, 3), Conv(128, 64, 1), Conv(64, 128, 3), MaxPool2d(2, 2))
        self.stage_4 = nn.Sequential(Conv(128, 256, 3), Conv(256, 128, 1), Conv(128, 256, 3), MaxPool2d(2, 2))
        self.stage_5 = nn.Sequential(Conv(256, 512, 3), Conv(512, 256, 1), Conv(256, 512, 3), Conv(512, 256, 1),
                                     Conv(256, 512, 3), MaxPool2d(2, 2))

        self.neck_5 = nn.Sequential(Conv(512, 256, 3), Conv(256, 512, 1), Conv(512, 256, 3), Conv(256, 512, 1),
                                    Conv(512, 256, 3))
        self.neck_4 = nn.Sequential(Conv(512, 256, 1), Conv(256, 512, 3), Conv(512, 256, 1), Conv(256, 512, 3),
                                    Conv(512, 256, 1))
        self.neck_3 = nn.Sequential(Conv(384, 256, 1), Conv(256, 512, 3), Conv(512, 256, 1), Conv(256, 512, 3),
                                    Conv(512, 256, 1))

        self.head2 = Conv(256, anchor_size, 1)
        self.head3 = Conv(256, anchor_size, 1)
        self.head4 = Conv(256, anchor_size, 1)

    def forward(self, x_0):
        x_1 = self.stage_1(x_0)
        x_2 = self.stage_2(x_1)
        x_3 = self.stage_3(x_2)
        x_4 = self.stage_4(x_3)
        x_5 = self.stage_5(x_4)

        xn_4_out = self.neck_5(x_5)
        xn_4 = F.interpolate(xn_4_out, scale_factor=2.0, mode='bilinear', align_corners=False)
        xn_3_out = self.neck_4(torch.cat([xn_4, x_4], dim=1))
        xn_3 = F.interpolate(xn_3_out, scale_factor=2.0, mode='bilinear', align_corners=False)
        xn_2_out = self.neck_3(torch.cat([xn_3, x_3], dim=1))

        x_2_out = self.head2(xn_2_out)
        x_3_out = self.head3(xn_3_out)
        x_4_out = self.head4(xn_4_out)
        return x_2_out, x_3_out, x_4_out


if __name__ == '__main__':
    net = DarkNet()
    x = torch.randn(2, 3, 640, 640)
    y = net(x)
    print(y.size())
