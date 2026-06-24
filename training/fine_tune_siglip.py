from __future__ import annotations

import argparse
import json
import random
import shutil
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from torch.utils.data import Dataset
from transformers import (
    AutoImageProcessor,
    AutoModelForImageClassification,
    Trainer,
    TrainingArguments,
)


LABELS = {"real": 0, "ai": 1}
ID2LABEL = {0: "real", 1: "ai"}


class ImageFolderDataset(Dataset):
    def __init__(self, files: list[tuple[Path, int]], processor: AutoImageProcessor):
        self.files = files
        self.processor = processor

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        path, label = self.files[index]
        image = Image.open(path).convert("RGB")
        item = self.processor(images=image, return_tensors="pt")
        return {
            "pixel_values": item["pixel_values"][0],
            "labels": torch.tensor(label, dtype=torch.long),
        }


def collect_files(data_dir: Path) -> list[tuple[Path, int]]:
    files: list[tuple[Path, int]] = []
    for label_name, label_id in LABELS.items():
        folder = data_dir / label_name
        if not folder.exists():
            raise SystemExit(f"Missing folder: {folder}")
        for path in folder.rglob("*"):
            if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}:
                files.append((path, label_id))
    if len(files) < 100:
        raise SystemExit("Training needs more images. Add curated images under data/real and data/ai.")
    return files


def split_files(files: list[tuple[Path, int]], seed: int, val_fraction: float) -> tuple[list[tuple[Path, int]], list[tuple[Path, int]]]:
    grouped = {0: [], 1: []}
    for item in files:
        grouped[item[1]].append(item)
    rng = random.Random(seed)
    train: list[tuple[Path, int]] = []
    val: list[tuple[Path, int]] = []
    for label_files in grouped.values():
        rng.shuffle(label_files)
        cut = max(1, int(len(label_files) * val_fraction))
        val.extend(label_files[:cut])
        train.extend(label_files[cut:])
    rng.shuffle(train)
    rng.shuffle(val)
    return train, val


def metrics(eval_pred):
    logits, labels = eval_pred
    probs = torch.softmax(torch.tensor(logits), dim=-1).numpy()[:, 1]
    preds = np.argmax(logits, axis=-1)
    output = {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds),
    }
    if len(set(labels.tolist() if hasattr(labels, "tolist") else labels)) == 2:
        output["roc_auc"] = roc_auc_score(labels, probs)
    return output


def export_onnx(model_dir: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    processor = AutoImageProcessor.from_pretrained(model_dir)
    model = AutoModelForImageClassification.from_pretrained(model_dir)
    model.eval()
    dummy = torch.randn(1, 3, 224, 224)
    onnx_path = output_dir / "model.onnx"
    torch.onnx.export(
        model,
        (dummy,),
        onnx_path,
        input_names=["pixel_values"],
        output_names=["logits"],
        dynamic_axes={"pixel_values": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=17,
    )
    processor.save_pretrained(output_dir)
    manifest = {
        "model": "model.onnx",
        "model_version": "custom-finetuned-siglip",
        "external_data": [],
        "total_size": onnx_path.stat().st_size,
    }
    (output_dir / "model-manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True, help="Folder with real/ and ai/ subfolders.")
    parser.add_argument("--base-model", default="Ateeqq/ai-vs-human-image-detector")
    parser.add_argument("--output-dir", type=Path, default=Path("training/runs/siglip-finetuned"))
    parser.add_argument("--site-export-dir", type=Path, default=None, help="Optional folder to receive exported ONNX files.")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    files = collect_files(args.data_dir)
    train_files, val_files = split_files(files, args.seed, 0.12)
    processor = AutoImageProcessor.from_pretrained(args.base_model)
    train_dataset = ImageFolderDataset(train_files, processor)
    val_dataset = ImageFolderDataset(val_files, processor)
    model = AutoModelForImageClassification.from_pretrained(
        args.base_model,
        num_labels=2,
        id2label=ID2LABEL,
        label2id=LABELS,
        ignore_mismatched_sizes=True,
    )

    training_args = TrainingArguments(
        output_dir=str(args.output_dir),
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        remove_unused_columns=False,
        seed=args.seed,
        fp16=torch.cuda.is_available(),
        report_to=[],
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=metrics,
    )
    trainer.train()
    trainer.save_model(args.output_dir / "best")
    processor.save_pretrained(args.output_dir / "best")

    export_dir = args.site_export_dir or (args.output_dir / "onnx-export")
    export_onnx(args.output_dir / "best", export_dir)
    if args.site_export_dir:
        for filename in ["model.onnx", "model-manifest.json"]:
            shutil.copy2(export_dir / filename, args.site_export_dir / filename)


if __name__ == "__main__":
    main()
