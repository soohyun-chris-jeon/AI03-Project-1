# src/submit.py

import datetime
import json
import os
import re
from pathlib import Path

import albumentations as A

# 다른 src 모듈 import
import config
import cv2  # OpenCV - 고급 이미지/비디오 처리
import pandas as pd
import torch
import utils
from albumentations.pytorch import ToTensorV2
from data.dataset import PillDataset
from model import get_model
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import get_unique_filepath

# --- 설정 ---
MODEL_CHECKPOINT = "./experiments/best_model.pt"  # 사용할 모델 경로

OUTPUT_DIR = Path("./output")  # AI03-Project-1/output 폴더에 저장됨
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

today_str = datetime.date.today().strftime("%y%m%d")
SUBMISSION_FILE = OUTPUT_DIR / f"submission_{today_str}.csv"  # submission_250723.csv


def extract_image_id(filename: str) -> int:
    """
    파일 이름에서 숫자 ID를 추출하는 함수.
    예: 'test_123.jpg' -> 123
    대회 데이터 파일 이름 규칙에 맞게 수정이 필요할 수 있음.
    """
    try:
        # 파일 이름에서 모든 숫자를 찾아 합친 후 정수로 변환
        return int("".join(re.findall(r"\d+", filename)))
    except:
        # 숫자 변환 실패 시 해시값 등 고유한 값으로 대체 (예외 처리)
        return hash(filename)


def main():
    print("--- Prediction Start ---")

    # 1. 라벨 맵 로드 (모델이 예측한 idx를 원래 category_id로 되돌리기 위함)
    label_map_path = Path(config.PROCESSED_DATA_DIR) / "label_map.json"
    with open(label_map_path, "r", encoding="utf-8") as f:
        label_map = json.load(f)
    # {'0': 101, '1': 542, ...} 형태의 딕셔너리가 필요
    idx_to_id = {int(k): v for k, v in label_map["idx_to_id"].items()}
    # --- 2. 데이터 준비 ---
    # NUM_CLASSES를 label_map을 통해 동적으로 설정 (배경 포함)
    NUM_CLASSES = len(label_map["id_to_idx"]) + 1

    # 2. 모델 로드
    model = get_model(num_classes=NUM_CLASSES).to(config.DEVICE)
    model.load_state_dict(torch.load(MODEL_CHECKPOINT, map_location=config.DEVICE))
    model.eval()
    print(f"Model loaded from {MODEL_CHECKPOINT}")

    # 3. 테스트 데이터 준비
    test_image_dir = "./data/raw/ai03-level1-project/test_images"  # 테스트 이미지 경로
    test_df = pd.DataFrame({"file_name": os.listdir(test_image_dir)})

    test_transforms = A.Compose(
        [
            A.Resize(512, 512),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ]
    )

    test_dataset = PillDataset(
        df=test_df, image_dir=test_image_dir, mode="test", transforms=test_transforms
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        collate_fn=utils.collate_fn,
    )

    # 4. 추론 및 결과 저장
    results = []
    with torch.no_grad():
        for images, image_filenames in tqdm(test_loader, desc="Predicting"):
            images = list(image.to(config.DEVICE) for image in images)
            outputs = model(images)

            for i, output in enumerate(outputs):
                image_filename = image_filenames[i]

                original_image_path = Path(test_image_dir) / image_filename
                original_image = cv2.imread(str(original_image_path))
                original_h, original_w, _ = original_image.shape

                # 리사이즈된 크기 (우리 코드에서는 512x512)
                resized_h, resized_w = 512, 512

                image_id = extract_image_id(image_filename)

                boxes = output["boxes"].cpu().numpy()
                labels = output["labels"].cpu().numpy()
                scores = output["scores"].cpu().numpy()

                # 각 box에 대해 결과 저장
                for box, label, score in zip(boxes, labels, scores):
                    # score 임계값 설정 (예: 0.5 이상만 저장)
                    if score < 0.5:
                        continue

                    xmin, ymin, xmax, ymax = box

                    # ---  2. bbox 좌표를 원본 크기로 스케일링 ---
                    xmin_orig = (xmin / resized_w) * original_w
                    ymin_orig = (ymin / resized_h) * original_h
                    xmax_orig = (xmax / resized_w) * original_w
                    ymax_orig = (ymax / resized_h) * original_h

                    # bbox 형식 변환: [xmin, ymin, xmax, ymax] -> [x, y, w, h]
                    bbox_x = xmin_orig
                    bbox_y = ymin_orig
                    bbox_w = xmax_orig - xmin_orig
                    bbox_h = ymax_orig - ymin_orig

                    # 라벨 복원: 0,1,2... -> 원래 category_id
                    category_id = idx_to_id[label]

                    results.append(
                        {
                            "image_id": image_id,
                            "category_id": category_id,
                            "bbox_x": bbox_x,
                            "bbox_y": bbox_y,
                            "bbox_w": bbox_w,
                            "bbox_h": bbox_h,
                            "score": score,
                        }
                    )

    # 5. 제출 파일 생성
    submission_df = pd.DataFrame(results)

    # annotation_id 추가 (1부터 시작하는 고유 ID)
    submission_df["annotation_id"] = range(1, len(submission_df) + 1)

    # 컬럼 순서 정리
    submission_df = submission_df[
        [
            "annotation_id",
            "image_id",
            "category_id",
            "bbox_x",
            "bbox_y",
            "bbox_w",
            "bbox_h",
            "score",
        ]
    ]

    # ⭐ 2. 파일을 저장하기 직전에, 중복되지 않는 최종 경로를 얻어옴
    unique_submission_path = get_unique_filepath(SUBMISSION_FILE)

    # ⭐ 3. 최종 경로에 파일 저장
    submission_df.to_csv(unique_submission_path, index=False)

    print("\n--- Prediction Finished! ---")
    # ⭐ 4. 저장된 실제 파일 경로를 출력
    print(f"Submission file saved to '{unique_submission_path}'")
    print(submission_df.head())


if __name__ == "__main__":
    main()
