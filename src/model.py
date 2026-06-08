import torch
import torch.nn as nn


class DeepPhys_HR(nn.Module):
    """
    DeepPhys cải biên để dự đoán Heart Rate (bpm) duy nhất.
    Giữ nguyên motion branch và appearance branch, sau đó dùng FC layers để hồi quy HR.
    """
    def __init__(self, in_channels=3, clip_length=64):
        super().__init__()
        self.clip_length = clip_length

        # ── Motion branch ──
        self.motion_conv1 = nn.Conv2d(in_channels, 16, kernel_size=3, padding=1)
        self.motion_bn1   = nn.BatchNorm2d(16)
        self.motion_conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.motion_bn2   = nn.BatchNorm2d(32)

        # ── Appearance branch ──
        self.appear_conv1 = nn.Conv2d(in_channels, 16, kernel_size=3, padding=1)
        self.appear_bn1   = nn.BatchNorm2d(16)
        self.appear_conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.appear_bn2   = nn.BatchNorm2d(32)

        # ── Global pooling + FC ──
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(64, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x, mask):
        """
        x:    (B, C, T, H, W)
        mask: (B, 1, T, H, W)
        Trả về: (B, 1) – nhịp tim dự đoán (bpm)
        """
        # Triệt tiêu vùng miệng
        x = x * mask

        B, C, T, H, W = x.shape
        # Reshape thành (B*T, C, H, W) để áp dụng 2D CNN
        x = x.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)

        # ── Motion branch (difference frame) ──
        x_prev = torch.empty_like(x)
        x_prev[0] = x[0]
        for i in range(1, B * T):
            if (i % T) == 0:          # đầu sequence mới → diff = 0
                x_prev[i] = x[i]
            else:
                x_prev[i] = x[i - 1]
        diff = x - x_prev

        m = self.relu(self.motion_bn1(self.motion_conv1(diff)))
        m = self.relu(self.motion_bn2(self.motion_conv2(m)))
        m = self.global_pool(m).view(B * T, 32)

        # ── Appearance branch ──
        a = self.relu(self.appear_bn1(self.appear_conv1(x)))
        a = self.relu(self.appear_bn2(self.appear_conv2(a)))
        a = self.global_pool(a).view(B * T, 32)

        # ── Fusion ──
        feat = torch.cat([m, a], dim=1)                 # (B*T, 64)

        # Pool theo thời gian để lấy 1 vector cho toàn clip
        feat = feat.view(B, T, 64)
        feat = feat.mean(dim=1)                         # (B, 64)

        # ── Regression Head ──
        hr = self.relu(self.fc1(feat))
        hr = self.dropout(hr)
        hr = self.relu(self.fc2(hr))
        hr = self.fc3(hr)                               # (B, 1)

        return hr.squeeze(-1)                           # (B,)