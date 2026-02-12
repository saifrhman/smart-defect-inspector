from __future__ import annotations

from torch.utils.data import DataLoader
from src.segmentation.dataset import FijiSegmentationDataset


def main() -> None:
    ds = FijiSegmentationDataset("data/segmentation/images", "data/segmentation/masks", size=256)
    dl = DataLoader(ds, batch_size=4, shuffle=True)

    x, y = next(iter(dl))
    print("batch image:", x.shape, x.dtype, x.min().item(), x.max().item())
    print("batch mask :", y.shape, y.dtype, y.min().item(), y.max().item())


if __name__ == "__main__":
    main()
