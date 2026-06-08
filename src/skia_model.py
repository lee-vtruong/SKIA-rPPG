import torch
import torch.nn as nn


class SKIANetLite(nn.Module):
    def __init__(self, in_channels=3):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv3d(in_channels, 16, kernel_size=(3, 5, 5), padding=(1, 2, 2)),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1, 2, 2)),

            nn.Conv3d(16, 32, kernel_size=(3, 3, 3), padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1, 2, 2)),

            nn.Conv3d(32, 64, kernel_size=(3, 3, 3), padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
        )

        self.temporal = nn.Sequential(
            nn.Conv1d(64, 64, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, 32, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv1d(32, 1, kernel_size=1),
        )

    def forward(self, x):
        # x: B,C,T,H,W
        feat = self.encoder(x)

        # spatial average pooling
        feat = feat.mean(dim=[3, 4])  # B,64,T

        out = self.temporal(feat).squeeze(1)  # B,T

        return out


def neg_pearson_loss(pred, target):
    pred = pred - pred.mean(dim=1, keepdim=True)
    target = target - target.mean(dim=1, keepdim=True)

    pred = pred / (pred.std(dim=1, keepdim=True) + 1e-6)
    target = target / (target.std(dim=1, keepdim=True) + 1e-6)

    corr = (pred * target).mean(dim=1)

    return -corr.mean()
