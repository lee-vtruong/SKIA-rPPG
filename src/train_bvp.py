import os, sys
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, Subset
import numpy as np
from tqdm import tqdm
from src.dataloader_bvp import UBFCrPPG_BVP
from src.model_bvp import DeepPhys_BVP

BATCH_SIZE = 1
EPOCHS = 30
LR = 1e-4
CLIP_LENGTH = 64
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def negative_pearson(pred, target):
    pred = pred - pred.mean(dim=1, keepdim=True)
    target = target - target.mean(dim=1, keepdim=True)
    num = (pred * target).sum(dim=1)
    den = torch.sqrt((pred**2).sum(dim=1) * (target**2).sum(dim=1) + 1e-8)
    return (1 - num/den).mean()

# Data
dataset = UBFCrPPG_BVP(root_dir='/raid/hvtham/whale/SIKA/data/UBFC-rPPG/1',
                       clip_length=CLIP_LENGTH, bvp_row=0)

# Tách test set
test_subs = ['subject41','subject42','subject43','subject44',
             'subject45','subject46','subject47','subject48']
test_idx = [i for i,s in enumerate(dataset.subjects) if s in test_subs]
train_val_idx = [i for i,s in enumerate(dataset.subjects) if s not in test_subs]
test_ds = Subset(dataset, test_idx)
train_val_ds = Subset(dataset, train_val_idx)

n_val = max(1, int(0.2*len(train_val_ds)))
n_train = len(train_val_ds) - n_val
train_ds, val_ds = random_split(train_val_ds, [n_train, n_val])

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
test_loader  = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

print(f'Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}')

# Model
model = DeepPhys_BVP(clip_length=CLIP_LENGTH).to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr=LR)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=4)

best_val_loss = float('inf')
for epoch in range(EPOCHS):
    model.train()
    train_loss = 0
    for frames, masks, bvp_gt in tqdm(train_loader, desc=f'Epoch {epoch+1}/{EPOCHS} [Train]'):
        frames = frames.to(DEVICE)
        masks = masks.to(DEVICE)
        bvp_gt = bvp_gt.to(DEVICE)
        bvp_pred = model(frames, masks)
        loss = negative_pearson(bvp_pred, bvp_gt)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    train_loss /= len(train_loader)

    model.eval()
    val_loss = 0
    with torch.no_grad():
        for frames, masks, bvp_gt in val_loader:
            frames = frames.to(DEVICE)
            masks = masks.to(DEVICE)
            bvp_gt = bvp_gt.to(DEVICE)
            bvp_pred = model(frames, masks)
            loss = negative_pearson(bvp_pred, bvp_gt)
            val_loss += loss.item()
    val_loss /= len(val_loader)
    scheduler.step(val_loss)

    print(f'Epoch {epoch+1}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}')
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), 'best_model_bvp.pth')
        print('   ✓ Saved')

# Đánh giá trên test set
model.load_state_dict(torch.load('best_model_bvp.pth'))
model.eval()
preds, targets = [], []
with torch.no_grad():
    for frames, masks, bvp_gt in test_loader:
        frames = frames.to(DEVICE)
        masks = masks.to(DEVICE)
        bvp_gt = bvp_gt.to(DEVICE)
        bvp_pred = model(frames, masks).squeeze(0).cpu()
        preds.append(bvp_pred)
        targets.append(bvp_gt.squeeze(0).cpu())

preds = torch.cat(preds).numpy()
targets = torch.cat(targets).numpy()
pearson = np.corrcoef(preds, targets)[0,1]
print(f'\nTest Pearson: {pearson:.4f}')