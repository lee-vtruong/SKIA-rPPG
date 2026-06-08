import torch
import torch.nn as nn

class DeepPhys_BVP(nn.Module):
    def __init__(self, in_channels=3, clip_length=64):
        super().__init__()
        self.clip_length = clip_length

        # Motion branch
        self.motion_conv1 = nn.Conv2d(in_channels, 16, 3, padding=1)
        self.motion_bn1   = nn.BatchNorm2d(16)
        self.motion_conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.motion_bn2   = nn.BatchNorm2d(32)

        # Appearance branch
        self.appear_conv1 = nn.Conv2d(in_channels, 16, 3, padding=1)
        self.appear_bn1   = nn.BatchNorm2d(16)
        self.appear_conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.appear_bn2   = nn.BatchNorm2d(32)

        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(64, 64)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x, mask):
        x = x * mask
        B, C, T, H, W = x.shape
        x = x.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)

        # Motion diff
        x_prev = x.roll(shifts=1, dims=0)
        diff = x - x_prev
        diff[::T] = 0.0   # reset diff for first frame of each sequence

        m = torch.relu(self.motion_bn1(self.motion_conv1(diff)))
        m = torch.relu(self.motion_bn2(self.motion_conv2(m)))
        m = self.global_pool(m).view(B * T, 32)

        a = torch.relu(self.appear_bn1(self.appear_conv1(x)))
        a = torch.relu(self.appear_bn2(self.appear_conv2(a)))
        a = self.global_pool(a).view(B * T, 32)

        feat = torch.cat([m, a], dim=1)        # (B*T, 64)
        feat = torch.relu(self.fc1(feat))
        out = self.fc2(feat).view(B, T)        # (B, T)
        return out