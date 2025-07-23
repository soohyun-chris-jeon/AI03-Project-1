# ====================================================================
# train.py
# ====================================================================
# dataset.py에서 데이터를 가져오고,
# model.py에서 모델을 불러와서 실제 학습(training)을 실행하는 스크립트.
# src/train.py

import gc
import json

# 1. 다른 모듈에서 필요한 모든 구성요소 import
import config
import pandas as pd
import torch
from model import get_model
from sklearn.model_selection import StratifiedGroupKFold
from src.assets.dataset import PillDataset
from src.assets.transforms import get_train_transforms, get_val_transforms
from torch.utils.data import DataLoader
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from tqdm import tqdm
from utils import collate_fn, seed_everything


def train():
    """메인 학습 및 검증 함수"""

    # --- 1. 기본 설정 ---
    seed_everything(config.SEED)

    # --- 2. 데이터 준비 ---
    df = pd.read_csv(config.SAVE_PATH)
    with open(config.LABEL_MAP_PATH, "r") as f:
        label_map = json.load(f)

    # NUM_CLASSES를 label_map을 통해 동적으로 설정 (배경 포함)
    num_classes = len(label_map["id_to_idx"]) + 1

    # StratifiedGroupKFold를 사용한 데이터 분할
    groups = df["file_name"]
    labels = df["category_id"]
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=config.SEED)
    train_idxs, val_idxs = next(cv.split(df, labels, groups))

    train_df_split = df.iloc[train_idxs]
    val_df_split = df.iloc[val_idxs]

    # --- 3. 데이터셋 및 데이터로더 생성 ---
    train_dataset = PillDataset(
        df=train_df_split,
        image_dir=config.TRAIN_IMAGE_DIR,
        mode="train",
        transforms=get_train_transforms(),
    )
    val_dataset = PillDataset(
        df=val_df_split,
        image_dir=config.TRAIN_IMAGE_DIR,
        mode="val",
        transforms=get_val_transforms(),
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=4,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=4,
    )

    # --- 4. 모델, 옵티마이저, 스케줄러, Metric 정의 ---
    model = get_model(num_classes, config.MODEL_NAME, config.USE_PRETRAINED)
    model.to(config.DEVICE)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        params, lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.NUM_EPOCHS, eta_min=1e-6
    )
    metric = MeanAveragePrecision(box_format="xyxy").to(config.DEVICE)

    # --- 5. 학습 루프 ---
    print("--- Start Training ---")
    for epoch in range(config.NUM_EPOCHS):
        # Training
        model.train()
        running_loss = 0.0
        loop = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{config.NUM_EPOCHS}]")

        for images, targets in loop:
            images = [img.to(config.DEVICE) for img in images]
            targets = [{k: v.to(config.DEVICE) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
            running_loss += losses.item()

        avg_train_loss = running_loss / len(train_loader)
        scheduler.step()

        # Validation
        model.eval()
        val_loss = 0.0
        metric.reset()
        with torch.no_grad():
            for images, targets in val_loader:
                images = [img.to(config.DEVICE) for img in images]
                targets = [
                    {k: v.to(config.DEVICE) for k, v in t.items()} for t in targets
                ]

                predictions = model(images)
                metric.update(predictions, targets)

        mAP_dict = metric.compute()
        print(
            f"Epoch {epoch+1} | Train Loss: {avg_train_loss:.4f} | Validation mAP@.50: {mAP_dict['map_50']:.4f}"
        )
        gc.collect()
        torch.cuda.empty_cache()

    print("--- Finish Training ---")


if __name__ == "__main__":
    # 데이터 전처리가 먼저 실행되어야 함
    # import subprocess
    # subprocess.run(["python", "src/data/prepare_data.py"])

    train()
