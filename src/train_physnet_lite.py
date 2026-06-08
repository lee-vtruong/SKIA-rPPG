import os
import numpy as np
import torch

from tqdm import tqdm
from torch.utils.data import DataLoader

from skia_dataset import SKIADataset
from physnet_lite import (
    PhysNetLite,
    neg_pearson_loss
)


def evaluate(model, loader, device):

    model.eval()

    corrs = []

    with torch.no_grad():

        for x, y in tqdm(
            loader,
            desc="eval",
            leave=False
        ):

            x = x.to(device)
            y = y.to(device)

            pred = model(x)

            pred_np = pred.cpu().numpy()
            y_np = y.cpu().numpy()

            for p, t in zip(pred_np, y_np):

                c = np.corrcoef(p, t)[0,1]

                if not np.isnan(c):
                    corrs.append(c)

    return float(np.mean(corrs))


def main():

    data_root = (
        "/raid/hvtham/whale/SIKA/data/UBFC-rPPG/1"
    )

    subjects = sorted(os.listdir(data_root))

    train_subjects = subjects[:32]
    test_subjects = subjects[32:]

    train_set = SKIADataset(
        data_root,
        train_subjects,
        clip_len=64,
        stride=128,
        size=96,
        mode="raw",
    )

    test_set = SKIADataset(
        data_root,
        test_subjects,
        clip_len=64,
        stride=128,
        size=96,
        mode="raw",
    )

    train_loader = DataLoader(
        train_set,
        batch_size=4,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_set,
        batch_size=4,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    model = PhysNetLite(
        in_channels=3
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-4,
        weight_decay=1e-4,
    )

    best_corr = -999

    os.makedirs(
        "../checkpoints",
        exist_ok=True
    )

    for epoch in range(1, 9):

        model.train()

        losses = []

        for x, y in tqdm(
            train_loader,
            desc=f"Epoch {epoch:03d} train"
        ):

            x = x.to(device)
            y = y.to(device)

            pred = model(x)

            loss = neg_pearson_loss(
                pred,
                y
            )

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

            losses.append(loss.item())

        test_corr = evaluate(
            model,
            test_loader,
            device
        )

        print(
            f"Epoch {epoch:03d} | "
            f"loss={np.mean(losses):.4f} | "
            f"test_corr={test_corr:.4f}"
        )

        if test_corr > best_corr:

            best_corr = test_corr

            torch.save(
                model.state_dict(),
                "../checkpoints/physnet_lite_best.pt"
            )

            print(
                "Saved best checkpoint"
            )

    print(
        f"\nBest PhysNet-lite "
        f"test corr: {best_corr:.4f}"
    )


if __name__ == "__main__":
    main()