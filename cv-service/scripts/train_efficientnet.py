"""
Training script cho EfficientNet-B4 food classifier.

Muc tieu: tao file weights/efficientnet_food.pt voi 291 classes
  - 101 classes Food-101 (mon Tay)
  - 190 classes nguyen lieu Viet Nam

Cau truc dataset:
  data/
    train/
      pizza/          <- ten folder = label (khop ALL_LABELS trong classifier.py)
        img001.jpg
      pho/
        img001.jpg
      ca_loc/
        img001.jpg
    val/
      pizza/
        img050.jpg

Chay:
  cd cv-service
  python scripts/train_efficientnet.py --data data/ --epochs 30 --batch 32

Tiep tuc tu checkpoint:
  python scripts/train_efficientnet.py --data data/ --resume weights/efficientnet_food.pt
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets
import torchvision.transforms as T
import timm
from tqdm import tqdm

from app.stages.classification.classifier import ALL_LABELS

DEFAULT_LR     = 1e-4
DEFAULT_EPOCHS = 30
DEFAULT_BATCH  = 32
PATIENCE       = 5
IMAGE_SIZE     = 380      # phai khop voi _CLASSIFY_TRANSFORM trong classifier.py
NUM_CLASSES    = len(ALL_LABELS)  # 291

TRAIN_TRANSFORM = T.Compose([
    T.RandomResizedCrop(IMAGE_SIZE, scale=(0.7, 1.0)),
    T.RandomHorizontalFlip(),
    T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.1),
    T.RandomRotation(15),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

VAL_TRANSFORM = T.Compose([
    T.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


class _RemappedDataset(torch.utils.data.Dataset):
    """
    ImageFolder sort folder theo alphabet nen index cua no lech voi ALL_LABELS.
    Class nay remap label -> index dung trong ALL_LABELS.
    """
    def __init__(self, ds: datasets.ImageFolder, remap: dict[int, int]):
        self.ds = ds
        self.remap = remap

    def __len__(self) -> int:
        return len(self.ds)

    def __getitem__(self, idx: int):
        img, label = self.ds[idx]
        return img, self.remap.get(label, label)


def _build_remap(ds: datasets.ImageFolder) -> dict[int, int]:
    known = set(ALL_LABELS)
    unknown = set(ds.classes) - known
    if unknown:
        print(f"WARNING: {len(unknown)} folder khong co trong ALL_LABELS: {unknown}")
        print("Doi ten folder cho khop de server nhan dung label.")
    return {
        idx: ALL_LABELS.index(name)
        for name, idx in ds.class_to_idx.items()
        if name in known
    }


def build_model(num_classes: int, resume: str | None) -> nn.Module:
    model = timm.create_model(
        "efficientnet_b4",
        pretrained=(resume is None),  # ImageNet pretrained neu train tu dau
        num_classes=num_classes,
    )
    if resume:
        state = torch.load(resume, map_location="cpu")
        model.load_state_dict(state)
        print(f"Resumed from {resume}")
    return model


def validate(model: nn.Module, loader: DataLoader, criterion, device: str) -> tuple[float, float]:
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            total_loss += criterion(outputs, labels).item() * len(labels)
            correct += (outputs.argmax(1) == labels).sum().item()
            total += len(labels)
    return total_loss / total, correct / total


def train(args: argparse.Namespace) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}  |  Classes: {NUM_CLASSES}")

    train_dir = Path(args.data) / "train"
    val_dir   = Path(args.data) / "val"

    if not train_dir.exists():
        print(f"ERROR: {train_dir} khong ton tai.")
        print("Tao cau truc: data/train/<label>/anh.jpg  va  data/val/<label>/anh.jpg")
        sys.exit(1)

    train_ds = datasets.ImageFolder(train_dir, transform=TRAIN_TRANSFORM)
    val_ds   = datasets.ImageFolder(val_dir,   transform=VAL_TRANSFORM)
    remap    = _build_remap(train_ds)

    print(f"Train: {len(train_ds)} anh | Val: {len(val_ds)} anh | "
          f"Classes tim thay: {len(train_ds.classes)}")

    train_loader = DataLoader(
        _RemappedDataset(train_ds, remap),
        batch_size=args.batch, shuffle=True,
        num_workers=args.workers, pin_memory=(device == "cuda"),
    )
    val_loader = DataLoader(
        _RemappedDataset(val_ds, remap),
        batch_size=args.batch, shuffle=False,
        num_workers=args.workers, pin_memory=(device == "cuda"),
    )

    model = build_model(NUM_CLASSES, args.resume)
    model.to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    # Backbone lr nho hon head 10x de tranh pha pretrained weights
    head_params     = list(model.classifier.parameters())
    backbone_params = [p for p in model.parameters()
                       if not any(p is h for h in head_params)]
    optimizer = torch.optim.AdamW([
        {"params": backbone_params, "lr": args.lr * 0.1},
        {"params": head_params,     "lr": args.lr},
    ], weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_val_acc  = 0.0
    patience_left = PATIENCE
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        run_loss, correct, total = 0.0, 0, 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}", leave=False)
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            run_loss += loss.item() * len(labels)
            correct  += (outputs.detach().argmax(1) == labels).sum().item()
            total    += len(labels)
            pbar.set_postfix(loss=f"{loss.item():.3f}", acc=f"{correct/total:.3f}")

        scheduler.step()
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        print(f"Epoch {epoch:>3} | "
              f"train_loss={run_loss/total:.4f}  train_acc={correct/total:.4f} | "
              f"val_loss={val_loss:.4f}  val_acc={val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc  = val_acc
            patience_left = PATIENCE
            torch.save(model.state_dict(), out_path)
            print(f"  [SAVED] {out_path}  (val_acc={val_acc:.4f})")
        else:
            patience_left -= 1
            print(f"  No improvement. Patience: {patience_left}/{PATIENCE}")
            if patience_left == 0:
                print("Early stopping.")
                break

    print(f"\nDone. Best val_acc={best_val_acc:.4f}")
    print(f"File da luu: {out_path}")
    print("Dat file vao weights/efficientnet_food.pt roi restart server.")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train EfficientNet-B4 food classifier")
    p.add_argument("--data",    required=True,
                   help="Thu muc goc chua train/ va val/")
    p.add_argument("--output",  default="weights/efficientnet_food.pt",
                   help="Duong dan luu file .pt")
    p.add_argument("--epochs",  type=int,   default=DEFAULT_EPOCHS)
    p.add_argument("--batch",   type=int,   default=DEFAULT_BATCH)
    p.add_argument("--lr",      type=float, default=DEFAULT_LR)
    p.add_argument("--workers", type=int,   default=2)
    p.add_argument("--resume",  default=None,
                   help="Tiep tuc tu checkpoint .pt co san")
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
