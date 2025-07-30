# ====================================================================
# ensemble.py
# ====================================================================
# 최종 아웃풋을 앙상블하는 코드
# src/ensemble.py

import datetime
import os
import re
from pathlib import Path

import cv2
import pandas as pd
from ensemble_boxes import weighted_boxes_fusion
from tqdm import tqdm

# --- 1. 설정 (모델 추가) ---
SUBMISSION_FILES = [
    "./output/submission_250729(2)_RESNET.csv",
    "./output/submission_YOLO.csv",
]
TEST_IMAGE_DIR = Path("./data/raw/ai03-level1-project/test_images")
today_str = datetime.date.today().strftime("%y%m%d")
ENSEMBLED_SUBMISSION_FILE = f"./output/submission_ensemble_{today_str}.csv"
IOU_THR = 0.55
SKIP_BOX_THR = 0.0001
WEIGHTS = [2, 1]


# predict.py에서 사용했던 id 추출 함수를 가져옴
def extract_image_id(filename: str) -> int:
    try:
        return int("".join(re.findall(r"\d+", filename)))
    except:
        return hash(filename)


def main():
    print("--- Ensemble Start ---")

    # --- 2. 데이터 준비 ---
    subs = [pd.read_csv(file) for file in SUBMISSION_FILES]
    combined_df = pd.concat(subs, ignore_index=True)

    # image_id와 실제 파일 이름을 미리 매핑
    test_files = os.listdir(TEST_IMAGE_DIR)
    # 실제 파일 이름을 기반으로 id를 추출하여 {id: filename} 딕셔너리를 생성
    id_to_filename_map = {extract_image_id(f): f for f in test_files}

    image_ids = combined_df["image_id"].unique()
    final_results = []

    # --- 3. WBF 핵심 로직 ---
    for image_id in tqdm(image_ids, desc="Ensembling predictions"):
        # 매핑된 딕셔너리에서 정확한 파일 이름을 가져옴
        image_filename = id_to_filename_map.get(image_id)
        if image_filename is None:
            print(
                f"Warning: Image file not found for image_id {image_id} in map. Skipping."
            )
            continue

        image_path = TEST_IMAGE_DIR / image_filename
        original_image = cv2.imread(str(image_path))
        h, w, _ = original_image.shape

        # ... (WBF를 위한 리스트 준비 및 채우는 로직은 동일) ...
        boxes_list, scores_list, labels_list = [], [], []
        for i in range(len(SUBMISSION_FILES)):
            model_df = subs[i]
            model_image_df = model_df[model_df["image_id"] == image_id]
            if model_image_df.empty:
                boxes_list.append([])
                scores_list.append([])
                labels_list.append([])
                continue

            boxes = model_image_df[["bbox_x", "bbox_y", "bbox_w", "bbox_h"]].values
            boxes[:, 2] += boxes[:, 0]
            boxes[:, 3] += boxes[:, 1]
            boxes[:, [0, 2]] /= w
            boxes[:, [1, 3]] /= h

            boxes_list.append(boxes.tolist())
            scores_list.append(model_image_df["score"].values.tolist())
            labels_list.append(model_image_df["category_id"].values.tolist())

        fused_boxes, fused_scores, fused_labels = weighted_boxes_fusion(
            boxes_list,  # 첫 번째 인자: 박스 리스트
            scores_list,  # 두 번째 인자: 점수 리스트
            labels_list,  # 세 번째 인자: 라벨 리스트
            weights=WEIGHTS,
            iou_thr=IOU_THR,
            skip_box_thr=SKIP_BOX_THR,
        )

        # ... (결과를 final_results에 추가하는 로직은 동일) ...
        for box, score, label in zip(fused_boxes, fused_scores, fused_labels):
            x_min, y_min, x_max, y_max = box
            final_results.append(
                {
                    "image_id": image_id,
                    "category_id": int(label),
                    "bbox_x": x_min * w,
                    "bbox_y": y_min * h,
                    "bbox_w": (x_max - x_min) * w,
                    "bbox_h": (y_max - y_min) * h,
                    "score": score,
                }
            )

    # --- 4. 최종 제출 파일 생성 ---
    # 빈 결과에 대한 예외 처리
    if not final_results:
        print(
            "Warning: No predictions were generated after ensembling. Creating an empty submission file."
        )
        # 헤더만 있는 빈 파일 생성
        final_df = pd.DataFrame(
            columns=[
                "annotation_id",
                "image_id",
                "category_id",
                "bbox_x",
                "bbox_y",
                "bbox_w",
                "bbox_h",
                "score",
            ]
        )
    else:
        final_df = pd.DataFrame(final_results)
        # image_id 기준으로 정렬하는 것이 좋음
        final_df = final_df.sort_values(by="image_id").reset_index(drop=True)

    final_df["annotation_id"] = range(1, len(final_df) + 1)
    final_df = final_df[
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

    final_df.to_csv(ENSEMBLED_SUBMISSION_FILE, index=False)

    print("\n--- Ensemble Finished! ---")
    print(f"Ensembled submission file saved to '{ENSEMBLED_SUBMISSION_FILE}'")
    print(final_df.head())


if __name__ == "__main__":
    main()
