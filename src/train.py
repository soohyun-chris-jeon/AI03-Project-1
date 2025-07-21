# ====================================================================
# train.py
# ====================================================================
# dataset.py에서 데이터를 가져오고,
# model.py에서 모델을 불러와서 실제 학습(training)을 실행하는 스크립트.


# src/train.py

# 다른 src 파일에서 필요한 것들을 import
import os

import config
import pandas as pd
import torch
import utils

# Data Augmentation
from dataset import PillDataset, test_transforms, train_transforms
from model import get_model

# 싸이킷런 데이터 나누기
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from tqdm import tqdm


def main():
    # 시드 고정
    utils.seed_everything(config.SEED)

    # 1. 데이터 준비
    df = pd.read_csv(config.PROCESSED_TRAIN_CSV)
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=config.SEED)
    test_df = pd.DataFrame({"file_name": os.listdir(config.TEST_IMAGE_DIR)})

    # --- Data set ---
    train_dataset = PillDataset(
        df=train_df,
        image_dir=config.TRAIN_IMAGE_DIR,
        mode="train",
        transforms=train_transforms,
    )
    val_dataset = PillDataset(
        df=val_df,
        image_dir=config.TRAIN_IMAGE_DIR,
        mode="val",
        transforms=train_transforms,
    )
    test_dataset = PillDataset(
        df=test_df,
        image_dir=config.TEST_IMAGE_DIR,
        mode="test",
        transforms=test_transforms,
    )

    # --- Data Loader ---
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        collate_fn=utils.collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        collate_fn=utils.collate_fn,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        collate_fn=utils.collate_fn,
    )

    # 2.
    model = get_model(num_classes=config.NUM_CLASSES).to(config.DEVICE)

    params = [p for p in model.parameters() if p.requires_grad]

    optimizer = torch.optim.SGD(
        params,
        lr=config.LEARNING_RATE,
        momentum=config.MOMENTUM,
        weight_decay=config.WEIGHT_DECAY,
    )

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    # 3. 학습 루프

    # 3. 학습 루프
    print("--- Start Training ---")
    # early_stopping = EarlyStopping(patience=7, verbose=True, path=path_model)
    train_losses = []
    val_losses = []
    for epoch in range(config.NUM_EPOCHS):
        # --- Training Step ---
        model.train()
        running_loss = 0.0
        loop = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{config.NUM_EPOCHS}]")

        for images, targets in loop:
            images = list(image.to(config.DEVICE) for image in images)
            targets = [{k: v.to(config.DEVICE) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
            running_loss += losses.item()
        avg_train_loss = running_loss / len(train_loader)

        scheduler.step()

        # --- Validation Step ---
        model.eval()
        val_loss = 0.0
        all_predictions = []
        all_ground_truths = []

        # (2) Validation phase
        with torch.no_grad():
            for images, targets in val_loader:
                images = [img.to(config.DEVICE) for img in images]
                predictions = model(images)  # SSD는 eval 모드에서 prediction만 한다
                # 저장: 추론 결과와 Ground Truth
                all_predictions.extend(predictions)
                all_ground_truths.extend(targets)

                model.train()
                targets = [
                    {k: v.to(config.DEVICE) for k, v in t.items()} for t in targets
                ]

                loss_dict = model(
                    images, targets
                )  # val_loss 계산을 위해 잠시 train 모드
                model.eval()

                losses = sum(loss for loss in loss_dict.values())
                val_loss += losses.item()
            average_val_loss = val_loss / len(val_loader)

            val_losses.append(average_val_loss)

            # mAP 평가
            mAP = evaluate_model(all_predictions, all_ground_truths, config.NUM_CLASSES)
            print(
                f"Train Loss: {avg_train_loss:.4f}, Val Loss: {average_val_loss:.4f}, Validation mAP: {mAP:.4f}"
            )

        print(f"Epoch {epoch+1}/{config.NUM_EPOCHS}, Loss: {losses.item():.4f}")

    print("--- Finish Training ---"),


# 최종 모델 저장
# torch.save(model.state_dict(), f"{EXPERIMENT_DIR}/final_model.pt")

if __name__ == "__main__":
    main()
