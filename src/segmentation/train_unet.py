from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from src.segmentation.dataset import FijiSegmentationDataset
from src.segmentation.unet import UNetSmall
from src.db.log_run import create_run, log_metrics


@dataclass
class TrainConfig:
    size: int = 256
    batch: int = 8
    epochs: int = 5
    lr: float = 1e-3
    val_split: float = 0.1
    device: str = "cpu"


def dice_coeff_from_logits(logits: torch.Tensor, targets: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    preds = (probs > 0.5).float()
    targets = targets.float()

    inter = (preds * targets).sum(dim=(1, 2, 3))
    union = preds.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3))
    dice = (2 * inter + eps) / (union + eps)
    return dice.mean()


def main() -> None:
    cfg = TrainConfig()
    device = torch.device(cfg.device)

    ds = FijiSegmentationDataset("data/segmentation/images", "data/segmentation/masks", size=cfg.size)

    n_val = max(1, int(len(ds) * cfg.val_split))
    n_train = len(ds) - n_val
    train_ds, val_ds = random_split(ds, [n_train, n_val], generator=torch.Generator().manual_seed(42))

    train_loader = DataLoader(train_ds, batch_size=cfg.batch, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch, shuffle=False, num_workers=0)

    model = UNetSmall(base=32).to(device)
    bce = nn.BCEWithLogitsLoss()
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    out_dir = Path("outputs/unet")
    out_dir.mkdir(parents=True, exist_ok=True)

    run_id = create_run(
        task="segmentation-train",
        model_name="unet-small",
        dataset_name="NEU-DET (Fiji masks)",
        notes=f"epochs={cfg.epochs} batch={cfg.batch} size={cfg.size} lr={cfg.lr}",
    )

    best_val_dice = -1.0

    for epoch in range(1, cfg.epochs + 1):
        t0 = time.time()
        model.train()
        train_loss = 0.0
        train_dice = 0.0
        n_batches = 0

        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)

            logits = model(x)
            loss = bce(logits, y)

            opt.zero_grad()
            loss.backward()
            opt.step()

            train_loss += loss.item()
            with torch.no_grad():
                train_dice += float(dice_coeff_from_logits(logits, y).item())
            n_batches += 1

        train_loss /= max(1, n_batches)
        train_dice /= max(1, n_batches)

        # validation
        model.eval()
        val_loss = 0.0
        val_dice = 0.0
        v_batches = 0

        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device)
                y = y.to(device)
                logits = model(x)
                loss = bce(logits, y)
                val_loss += loss.item()
                val_dice += float(dice_coeff_from_logits(logits, y).item())
                v_batches += 1

        val_loss /= max(1, v_batches)
        val_dice /= max(1, v_batches)

        # save best
        if val_dice > best_val_dice:
            best_val_dice = val_dice
            torch.save(model.state_dict(), out_dir / "best.pt")

        dt = time.time() - t0
        print(f"Epoch {epoch}/{cfg.epochs} | train_loss={train_loss:.4f} dice={train_dice:.4f} "
              f"| val_loss={val_loss:.4f} dice={val_dice:.4f} | {dt:.1f}s")

        log_metrics(
            run_id,
            {
                f"train_loss_e{epoch}": train_loss,
                f"train_dice_e{epoch}": train_dice,
                f"val_loss_e{epoch}": val_loss,
                f"val_dice_e{epoch}": val_dice,
            },
        )

    print(f"Training complete. Best val dice: {best_val_dice:.4f}")
    print(f"Saved best weights: {(out_dir / 'best.pt').resolve()}")
    print(f"Logged run_id: {run_id}")


if __name__ == "__main__":
    main()
