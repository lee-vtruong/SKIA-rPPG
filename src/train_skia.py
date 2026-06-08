import os
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader

from skia_dataset import SKIADataset
from skia_model import SKIANetLite, neg_pearson_loss


def get_in_channels(mode):
    if mode == "raw":
        return 3
    if mode == "masked":
        return 3
    if mode == "raw_masked":
        return 6
    if mode == "full":
        return 9
    raise ValueError(mode)


def evaluate(model, loader, device):
    model.eval()
    corrs = []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)

            pred = model(x)

            pred_np = pred.cpu().numpy()
            y_np = y.cpu().numpy()

            for p, t in zip(pred_np, y_np):
                c = np.corrcoef(p, t)[0, 1]
                if not np.isnan(c):
                    corrs.append(c)

    return float(np.mean(corrs))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="raw",
                        choices=["raw", "masked", "raw_masked", "full"])
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--clip-len", type=int, default=128)
    parser.add_argument("--stride", type=int, default=64)
    parser.add_argument("--size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-4)
    args = parser.parse_args()

    data_root = "/raid/hvtham/whale/SIKA/data/UBFC-rPPG/1"

    subjects = sorted(os.listdir(data_root))

    train_subjects = subjects[:32]
    test_subjects = subjects[32:]

    print("Train subjects:", train_subjects)
    print("Test subjects:", test_subjects)

    train_set = SKIADataset(
        data_root,
        train_subjects,
        clip_len=args.clip_len,
        stride=args.stride,
        size=args.size,
        mode=args.mode,
    )

    test_set = SKIADataset(
        data_root,
        test_subjects,
        clip_len=args.clip_len,
        stride=args.stride,
        size=args.size,
        mode=args.mode,
    )

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = SKIANetLite(in_channels=get_in_channels(args.mode)).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    best_corr = -999

    os.makedirs("../checkpoints", exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []

        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)

            pred = model(x)

            loss = neg_pearson_loss(pred, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses.append(loss.item())

        test_corr = evaluate(model, test_loader, device)

        print(
            f"Epoch {epoch:03d} | "
            f"loss={np.mean(losses):.4f} | "
            f"test_corr={test_corr:.4f}"
        )

        if test_corr > best_corr:
            best_corr = test_corr
            save_path = f"../checkpoints/skia_{args.mode}_best.pt"
            torch.save(model.state_dict(), save_path)
            print(f"Saved best to {save_path}")

    print(f"Best test corr for mode={args.mode}: {best_corr:.4f}")


if __name__ == "__main__":
    main()
