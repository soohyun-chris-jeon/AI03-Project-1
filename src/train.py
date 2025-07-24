# ====================================================================
# train.py
# ====================================================================
# dataset.py에서 데이터를 가져오고,
# model.py에서 모델을 불러와서 실제 학습(training)을 실행하는 스크립트.
# src/train.py

import argparse  # argparse 라이브러리 import
import gc
import json

# 1. 다른 모듈에서 필요한 모든 구성요소 import
import config
import pandas as pd
import torch
from assets.dataset import PillDataset
from assets.transforms import get_train_transforms, get_val_transforms
from model import get_model
from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import DataLoader
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from torchvision.ops import box_iou
from tqdm import tqdm
from utils import EarlyStopping, collate_fn, seed_everything


def train(args):
    """메인 학습 및 검증 함수"""

    # ====================================================================
    # --- 1. 기본 설정 ---
    seed_everything(config.SEED)

    # ====================================================================
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

    # ====================================================================
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

    # ====================================================================
    # --- 4. 모델, 옵티마이저, 스케줄러, Metric 정의 ---
    model = get_model(num_classes, args.MODEL_NAME, config.USE_PRETRAINED)
    model.to(config.DEVICE)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        params, lr=args.LEARNING_RATE, weight_decay=args.WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.NUM_EPOCHS, eta_min=1e-6
    )

    # --- 5. 학습 루프 ---
    # ====================================================================
    metric = MeanAveragePrecision(box_format="xyxy").to(config.DEVICE)
    early_stopping = EarlyStopping(
        patience=5,
        verbose=True,
        mode="max",
        evaluation_name="Validation mAP",
        path="../experiments/" + args.MODEL_NAME + "pt",
    )

    train_losses = []
    val_losses = []
    print("--- Start Training ---")
    for epoch in range(args.NUM_EPOCHS):
        # Training
        model.train()
        running_loss = 0.0
        loop = tqdm(
            train_loader,
            desc=f"Epoch [{epoch+1}/{args.NUM_EPOCHS}]",
            dynamic_ncols=True,
        )

        for images, targets in loop:
            images = [img.to(config.DEVICE) for img in images]
            targets = [{k: v.to(config.DEVICE) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
            running_loss += losses.item()
            current_avg_loss = running_loss / (loop.n + 1)
            loop.set_postfix(loss=current_avg_loss)

        avg_train_loss = running_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # =====================================
        #  Validation Step
        # =====================================
        model.eval()
        val_loss = 0.0
        total_iou = 0
        tp_count = 0

        metric.reset()
        for images, targets in val_loader:
            images = [img.to(config.DEVICE) for img in images]
            targets = [{k: v.to(config.DEVICE) for k, v in t.items()} for t in targets]
            # 1. mAP 계산을 위한 예측 (그래디언트 계산 불필요)
            with torch.no_grad():  # 예측 부분만 no_grad로 감싸기
                predictions = model(images)

            metric.update(predictions, targets)

            # IoU 평가 메트릭
            for i in range(len(predictions)):
                pred_boxes = predictions[i]["boxes"]
                gt_boxes = targets[i]["boxes"]

                if len(pred_boxes) == 0 or len(gt_boxes) == 0:
                    continue

                # 각 GT 박스에 대해 가장 IoU가 높은 예측 박스를 찾음
                iou_matrix = box_iou(pred_boxes, gt_boxes)
                max_iou_per_gt, _ = iou_matrix.max(dim=0)

                # IoU가 0.5 이상인 경우 (TP)만 계산에 포함
                true_positives = max_iou_per_gt[max_iou_per_gt > 0.5]

                if len(true_positives) > 0:
                    total_iou += true_positives.sum().item()
                    tp_count += len(true_positives)
            model.train()  # Loss 계산을 위해 잠시 train 모드로
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            val_loss += losses.item()
            model.eval()  # 다음 배치를 위해 다시 eval 모드로 복귀

        average_val_loss = val_loss / len(val_loader)
        val_losses.append(average_val_loss)

        avg_iou = total_iou / tp_count if tp_count > 0 else 0.0

        # mAP 평가
        mAP_dict = metric.compute()

        # mAP50 기반EarlyStopping 로직 호출
        early_stopping(mAP_dict["map_50"], model)
        if early_stopping.early_stop:
            print("Early stopping")
            break

        # scheduler 설정
        scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]

        loop.write(
            f"Train Loss: {avg_train_loss:.4f}, Val Loss: {average_val_loss:.4f}, mAP@50: {mAP_dict['map_50']:.4f}, LR: {current_lr:.0e}"
        )
        loop.write(
            f"COCO mAP: {mAP_dict['map']:.4f}, mAP@75: {mAP_dict['map_75']:.4f}, Avg_IoU: {avg_iou:.4f}"
        )

        gc.collect()
        # torch.cuda.empty_cache()

    print("--- Finish Training ---")


if __name__ == "__main__":
    # 데이터 전처리가 먼저 실행되어야 함
    # import subprocess
    # subprocess.run(["python", "src/data/prepare_data.py"])

    parser = argparse.ArgumentParser(description="Object Detection Model Training")

    # 여기서 실험마다 바꾸고 싶은 파라미터를 정의
    parser.add_argument("--SEED", type=int, default=config.SEED, help="Random seed")
    parser.add_argument(
        "--NUM_EPOCHS",
        type=int,
        default=config.NUM_EPOCHS,
        help="Number of epochs to train",
    )
    parser.add_argument(
        "--MODEL_NAME", type=str, default=config.MODEL_NAME, help="Model name to use"
    )
    parser.add_argument(
        "--LEARNING_RATE",
        type=float,
        default=config.LEARNING_RATE,
        help="Learning rate",
    )
    parser.add_argument(
        "--WEIGHT_DECAY", type=float, default=config.WEIGHT_DECAY, help="Weight decay"
    )

    # 실험 관리를 위한 이름. wandb와 연동할 때 매우 유용함.
    parser.add_argument(
        "--run-name",
        type=str,
        default=config.RUN_NAME,
        help="Name for this experiment run",
    )

    args = parser.parse_args()

    # 3. 파싱된 인자(args)를 train 함수에 전달
    print("--- Starting Experiment with following configurations ---")
    print(args)
    train(args)
