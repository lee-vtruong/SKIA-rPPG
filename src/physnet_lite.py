import torch
import torch.nn as nn


class PhysNetLite(nn.Module):

    def __init__(self, in_channels=3):
        super().__init__()

        self.net = nn.Sequential(

            nn.Conv3d(
                in_channels,
                16,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True),

            nn.Conv3d(
                16,
                32,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),

            nn.MaxPool3d(
                kernel_size=(1,2,2)
            ),

            nn.Conv3d(
                32,
                64,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),

            nn.MaxPool3d(
                kernel_size=(1,2,2)
            ),

            nn.Conv3d(
                64,
                64,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
        )

        self.head = nn.Sequential(

            nn.Conv1d(
                64,
                64,
                kernel_size=7,
                padding=3
            ),

            nn.ReLU(inplace=True),

            nn.Conv1d(
                64,
                32,
                kernel_size=5,
                padding=2
            ),

            nn.ReLU(inplace=True),

            nn.Conv1d(
                32,
                1,
                kernel_size=1
            ),
        )

    def forward(self, x):

        # x: B,C,T,H,W

        feat = self.net(x)

        # B,64,T,H,W -> B,64,T
        feat = feat.mean(dim=[3,4])

        out = self.head(feat)

        return out.squeeze(1)


def neg_pearson_loss(pred, target):

    pred = pred - pred.mean(dim=1, keepdim=True)
    target = target - target.mean(dim=1, keepdim=True)

    pred = pred / (
        pred.std(dim=1, keepdim=True) + 1e-6
    )

    target = target / (
        target.std(dim=1, keepdim=True) + 1e-6
    )

    return -(pred * target).mean(dim=1).mean()