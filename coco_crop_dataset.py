import json
from pathlib import Path
from typing import List, Tuple

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image

class CocoCropDataset(Dataset):
    def __init__(
        self,
        ann_path: Path,
        img_dir: Path,
        transform=None
    ):
        self.img_dir = img_dir
        self.transform = transform

        with ann_path.open("r", encoding="utf-8") as f:
            coco = json.load(f)

        self.images = {im["id"]: im for im in coco["images"]}
        self.annotations = coco["annotations"]

        # Sort categories by ID for deterministic class order
        categories = sorted(coco["categories"], key=lambda c: c["id"])
        self.class_names = [c["name"] for c in categories]
        self.cat_id_to_idx = {c["id"]: i for i, c in enumerate(categories)}

        self.samples: List[Tuple[Path, Tuple[float, float, float, float], int]] = []

        for ann in self.annotations:
            img_info = self.images[ann["image_id"]]
            img_path = img_dir / Path(img_info["file_name"]).name
            if not img_path.exists():
                continue

            bbox = ann["bbox"]  # x, y, w, h
            cls = self.cat_id_to_idx[ann["category_id"]]

            self.samples.append((img_path, bbox, cls))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, bbox, cls = self.samples[idx]
        img = Image.open(img_path).convert("RGB")

        x, y, w, h = bbox

        # Convert to integer pixel coords
        x1 = int(round(x))
        y1 = int(round(y))
        x2 = int(round(x + w))
        y2 = int(round(y + h))

        # Clip to image boundaries
        x1 = max(0, min(x1, img.width - 1))
        y1 = max(0, min(y1, img.height - 1))
        x2 = max(0, min(x2, img.width))
        y2 = max(0, min(y2, img.height))

        # ---- CRITICAL FIX ----
        if x2 <= x1 or y2 <= y1:
            # Skip invalid crop by returning a minimal fallback
            return self.__getitem__((idx + 1) % len(self))
        else:
            crop = img.crop((x1, y1, x2, y2))

        if self.transform:
            crop = self.transform(crop)

        return crop, cls
