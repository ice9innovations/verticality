"""Reproducibility, device selection, and evaluation metrics."""

from __future__ import annotations

import random

import torch


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def seed_worker(worker_id: int) -> None:
    del worker_id
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)


def choose_device(requested: str) -> torch.device:
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(requested)


class ClassificationMetrics:
    def __init__(self) -> None:
        self.confusion = torch.zeros(4, 4, dtype=torch.long)
        self.correct_confidence: list[float] = []
        self.incorrect_confidence: list[float] = []

    def update(self, logits: torch.Tensor, targets: torch.Tensor) -> None:
        probabilities = logits.softmax(dim=1)
        confidence, predictions = probabilities.max(dim=1)
        for target, prediction in zip(targets.cpu(), predictions.cpu()):
            self.confusion[target, prediction] += 1
        matches = predictions.eq(targets)
        self.correct_confidence.extend(confidence[matches].detach().cpu().tolist())
        self.incorrect_confidence.extend(confidence[~matches].detach().cpu().tolist())

    def report(self) -> str:
        total = int(self.confusion.sum())
        correct = int(self.confusion.diag().sum())
        lines = [f"overall accuracy: {correct / max(total, 1):.4f}", "per-class accuracy:"]
        for label, degrees in enumerate((0, 90, 180, 270)):
            count = int(self.confusion[label].sum())
            lines.append(f"  {degrees:>3}°: {int(self.confusion[label, label]) / max(count, 1):.4f} ({count} samples)")
        lines.extend(("confusion matrix (rows=true, columns=predicted; 0/90/180/270):", str(self.confusion.tolist())))
        correct_conf = sum(self.correct_confidence) / len(self.correct_confidence) if self.correct_confidence else float("nan")
        wrong_conf = sum(self.incorrect_confidence) / len(self.incorrect_confidence) if self.incorrect_confidence else float("nan")
        lines.append(f"average confidence, correct: {correct_conf:.4f}")
        lines.append(f"average confidence, incorrect: {wrong_conf:.4f}")
        return "\n".join(lines)

