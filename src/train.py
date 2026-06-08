import os, sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, Subset
from tqdm import tqdm
import numpy as np

from src.dataloader import UBFCrPPG_HR
from src.model import DeepPhys_HR


# ══════════════════ Cấu hình ══════════════════
BATCH_SIZE   = 1
EPOCHS       = 30
LR           = 1e-4
CLIP_LENGTH  = 64
DEVICE       = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ══════════════════ Data ══════════════════
dataset = UBFCrPPG_HR(
    root_dir='/raid/hvtham/whale/SIKA/data/UBFC-rPPG/1',
    clip_length=CLIP_LENGTH,
    hr_row=1,
    fs=30
)

# Tách 8 subject làm test set độc lập
TEST_SUBS = ['subject41','subject42','subject43','subject44',
             'subject45','subject46','subject47','subject48']

test_indices = [i for i, s in enumerate(dataset.subjects) if s in TEST_SUBS]
train_val_indices = [i for i, s in enumerate(dataset.subjects) if s not in TEST_SUBS]

test_ds = Subset(dataset, test_indices)
train_val_ds = Subset(dataset, train_val_indices)

n_val = max(1, int(0.2 * len(train_val_ds)))
n_train = len(train_val_ds) - n_val
train_ds, val_ds = random_split(train_val_ds, [n_train, n_val])

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False)
test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False)

print(f'Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}')

# ══════════════════ Model ══════════════════
model = DeepPhys_HR(in_channels=3, clip_length=CLIP_LENGTH).to(DEVICE)
criterion = nn.L1Loss()          # MAE loss – trực tiếp tối ưu MAE
optimizer = optim.Adam(model.parameters(), lr=LR)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                  factor=0.5, patience=4)

best_val_loss = float('inf')

# ══════════════════ Training Loop ══════════════════
for epoch in range(EPOCHS):
    # ── Train ──
    model.train()
    train_loss = 0.0
    for frames, masks, hr_gt in tqdm(train_loader, desc=f'Epoch {epoch+1}/{EPOCHS} [Train]'):
        frames = frames.to(DEVICE)
        masks  = masks.to(DEVICE)
        hr_gt  = hr_gt.to(DEVICE)

        hr_pred = model(frames, masks)
        loss = criterion(hr_pred, hr_gt)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    train_loss /= len(train_loader)

    # ── Val ──
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for frames, masks, hr_gt in val_loader:
            frames = frames.to(DEVICE)
            masks  = masks.to(DEVICE)
            hr_gt  = hr_gt.to(DEVICE)

            hr_pred = model(frames, masks)
            loss = criterion(hr_pred, hr_gt)
            val_loss += loss.item()

    val_loss /= len(val_loader)
    scheduler.step(val_loss)

    print(f'Epoch {epoch+1:2d}: Train MAE={train_loss:.3f} bpm | Val MAE={val_loss:.3f} bpm')

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), 'best_model_hr.pth')
        print(f'   ✓ Model saved (val MAE={best_val_loss:.3f} bpm)')

print('\n' + '='*50)

# ══════════════════ Test Evaluation ══════════════════
model.load_state_dict(torch.load('best_model_hr.pth'))
model.eval()

preds, targets = [], []
with torch.no_grad():
    for frames, masks, hr_gt in test_loader:
        frames = frames.to(DEVICE)
        masks  = masks.to(DEVICE)
        hr_gt  = hr_gt.to(DEVICE)

        hr_pred = model(frames, masks)
        preds.append(hr_pred.cpu().numpy().flatten()[0])
        targets.append(hr_gt.cpu().numpy().flatten()[0])

preds   = np.array(preds)
targets = np.array(targets)

mae    = np.mean(np.abs(preds - targets))
rmse   = np.sqrt(np.mean((preds - targets)**2))
std    = np.std(preds - targets)

print('▸▸▸ Kết quả trên tập Test (8 subjects) ▸▸▸')
print(f'  MAE  = {mae:.2f} bpm')
print(f'  RMSE = {rmse:.2f} bpm')
print(f'  Std  = {std:.2f} bpm')

# Lưu kết quả
np.savez('hr_results.npz', preds=preds, targets=targets)
print('Đã lưu hr_results.npz')