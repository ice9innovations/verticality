#!/usr/bin/env python3
"""Train or evaluate the synthetic COCO orientation experiment."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import OrientationDataset
from model import build_model, load_model
from utils import ClassificationMetrics, choose_device, seed_everything, seed_worker


def loader(dataset, batch_size: int, workers: int, shuffle: bool, seed: int) -> DataLoader:
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=workers,
                      pin_memory=torch.cuda.is_available(), persistent_workers=workers > 0,
                      worker_init_fn=seed_worker, generator=generator)


def run_epoch(model, batches, device, criterion, optimizer=None):
    training = optimizer is not None
    model.train(training)
    total_loss = total_correct = total = 0
    metrics = ClassificationMetrics()
    context = torch.enable_grad() if training else torch.inference_mode()
    with context:
        for images, targets in tqdm(batches, desc="train" if training else "validate", leave=False):
            images, targets = images.to(device), targets.to(device)
            if training:
                optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, targets)
            if training:
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * targets.size(0)
            total_correct += logits.argmax(1).eq(targets).sum().item()
            total += targets.size(0)
            if not training:
                metrics.update(logits, targets)
    return total_loss / total, total_correct / total, metrics


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-dir", type=Path, help="COCO train2017 directory")
    parser.add_argument("--val-dir", type=Path, required=True, help="COCO val2017 directory")
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/best.pt"))
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, cuda:0, or mps")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    device = choose_device(args.device)
    print(f"device: {device}")
    criterion = nn.CrossEntropyLoss()
    if args.eval_only:
        model, checkpoint = load_model(args.checkpoint, device)
        image_size = int(checkpoint.get("image_size", args.image_size))
        val_data = OrientationDataset(args.val_dir, training=False, size=image_size)
        val_loader = loader(val_data, args.batch_size, args.workers, False, args.seed)
        loss, accuracy, metrics = run_epoch(model, val_loader, device, criterion)
        print(f"checkpoint epoch: {checkpoint.get('epoch', 'unknown')}  val loss: {loss:.4f}  val accuracy: {accuracy:.4f}")
        print(metrics.report())
        return
    if args.train_dir is None:
        raise SystemExit("--train-dir is required unless --eval-only is used")
    val_data = OrientationDataset(args.val_dir, training=False, size=args.image_size)
    val_loader = loader(val_data, args.batch_size, args.workers, False, args.seed)
    train_data = OrientationDataset(args.train_dir, training=True, size=args.image_size)
    train_loader = loader(train_data, args.batch_size, args.workers, True, args.seed)
    model = build_model(pretrained=True).to(device)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    best_accuracy = -1.0
    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, args.epochs + 1):
        train_loss, train_accuracy, _ = run_epoch(model, train_loader, device, criterion, optimizer)
        val_loss, val_accuracy, metrics = run_epoch(model, val_loader, device, criterion)
        print(f"epoch {epoch:02d}/{args.epochs}: train loss={train_loss:.4f} accuracy={train_accuracy:.4f} | "
              f"val loss={val_loss:.4f} accuracy={val_accuracy:.4f} lr={scheduler.get_last_lr()[0]:.2e}")
        print(metrics.report())
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            torch.save({"model_state": model.state_dict(), "epoch": epoch, "val_accuracy": val_accuracy,
                        "image_size": args.image_size, "classes": [0, 90, 180, 270]}, args.checkpoint)
            print(f"saved new best checkpoint: {args.checkpoint}")
        scheduler.step()


if __name__ == "__main__":
    main()
