# ====================================================================
# eval.py
# ====================================================================
# 학습 및 평가를 위한 소스 코드

# src/eval.py

import argparse
import json

import albumentations as A

# 다른 src 모듈 import
import config
import pandas as pd
import torch
import utils
from albumentations.pytorch import ToTensorV2
from assets.dataset import PillDataset
from model import get_model
from torch.utils.data import DataLoader
from torchmetrics.detection import MeanAveragePrecision
from tqdm import tqdm


def evaluate(args):
    """주어진 모델 체크포인트와 폴드에 대해 평가를 수행하는 함수"""

    print(f"--- Evaluating Fold {args.fold} using checkpoint {args.checkpoint} ---")

    # --- 1. 데이터 준비 ---
    df = pd.read_csv(config.PROCESSED_TRAIN_CSV)
    val_df = df[df["fold"] == args.fold].reset_index(drop=True)

    # 검증용 변환 (학습 시 사용했던 validation transform과 동일하게)
    val_transforms = A.Compose(
        [
            A.Resize(512, 512),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ],
        bbox_params=A.BboxParams(format="pascal_voc", label_fields=["labels"]),
    )

    val_dataset = PillDataset(
        df=val_df,
        image_dir=config.TRAIN_IMAGE_DIR,
        mode="val",
        transforms=val_transforms,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        collate_fn=utils.collate_fn,
    )

    # --- 2. 모델 로드 ---
    # label_map.json을 통해 정확한 클래스 수를 가져옴
    with open(config.LABEL_MAP_PATH, "r", encoding="utf-8") as f:
        label_map = json.load(f)
    num_classes = len(label_map["id_to_idx"])

    model = get_model(num_classes=num_classes).to(config.DEVICE)
    model.load_state_dict(torch.load(args.checkpoint, map_location=config.DEVICE))
    model.eval()

    # --- 3. 평가지표 객체 초기화 ---
    # box_format='xyxy'는 모델이 [xmin, ymin, xmax, ymax] 형식으로 bbox를 출력한다는 의미
    metric = MeanAveragePrecision(box_format="xyxy")

    # --- 4. 평가 루프 ---
    with torch.no_grad():
        for images, targets in tqdm(val_loader, desc=f"Evaluating Fold {args.fold}"):
            images = list(image.to(config.DEVICE) for image in images)

            # 모델 예측
            outputs = model(images)

            # torchmetrics가 CPU에서 계산하므로, 예측과 정답을 CPU로 이동
            preds = [{k: v.to("cpu") for k, v in t.items()} for t in outputs]
            targets = [{k: v.to("cpu") for k, v in t.items()} for t in targets]

            # 평가지표 업데이트
            metric.update(preds, targets)

    # --- 5. 최종 점수 계산 및 출력 ---
    final_scores = metric.compute()

    print("\n--- Evaluation Results ---")
    for key, value in final_scores.items():
        # 텐서 값을 일반 숫자로 변환하여 출력
        print(f"{key}: {value.item():.4f}")

    print(f"\n--- Evaluation for Fold {args.fold} Finished! ---")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a trained model.")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to the model checkpoint file.",
    )
    parser.add_argument(
        "--fold", type=int, required=True, help="Which fold to use for validation."
    )

    args = parser.parse_args()
    evaluate(args)
