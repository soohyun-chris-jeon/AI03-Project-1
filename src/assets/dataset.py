# ====================================================================
# dataset.py
# ====================================================================
# PillDataset 클래스 정의
# src/data/dataset.py

from pathlib import Path

import cv2
import torch
from torch.utils.data import Dataset


class PillDataset(Dataset):
    def __init__(self, df, image_dir, mode="train", transforms=None):
        self.df = df
        self.image_dir = Path(image_dir)
        self.mode = mode
        self.transforms = transforms
        self.image_ids = self.df["file_name"].unique()

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        image_path = self.image_dir / image_id

        image = cv2.imread(str(image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.mode in ["train", "val"]:
            records = self.df[self.df["file_name"] == image_id]
            boxes = records[["bbox_x", "bbox_y", "bbox_w", "bbox_h"]].values
            boxes[:, 2] = boxes[:, 0] + boxes[:, 2]
            boxes[:, 3] = boxes[:, 1] + boxes[:, 3]
            labels = records["label_idx"].values

            if self.transforms:
                transformed = self.transforms(image=image, bboxes=boxes, labels=labels)
                image = transformed["image"]
                boxes = transformed["bboxes"]
                labels = transformed["labels"]

            target = {
                "boxes": torch.as_tensor(boxes, dtype=torch.float32),
                "labels": torch.as_tensor(labels, dtype=torch.int64),
            }
            if len(target["boxes"]) == 0:
                target["boxes"] = torch.zeros((0, 4), dtype=torch.float32)

            return image, target

        elif self.mode == "test":
            if self.transforms:
                transformed = self.transforms(image=image)
                image = transformed["image"]
            return image, image_id
