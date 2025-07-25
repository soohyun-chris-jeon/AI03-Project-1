# src/ensemble.py

import pandas as pd
import numpy as np
import os
import cv2
from pathlib import Path
from tqdm import tqdm
from ensemble_boxes import weighted_boxes_fusion
import datetime


# --- 1. 설정 (모델 추가) ---
# 합치고 싶은 submission CSV 파일들의 경로 리스트
SUBMISSION_FILES = [
    './output/submission_faster_rcnn_250725.csv',
    './output/submission_yolo_250725.csv',
    # './output/submission_another_model_250725.csv', # 모델이 더 있다면 여기에 추가
]

# 원본 테스트 이미지들이 있는 폴더 경로 (좌표 정규화를 위해 필요)
TEST_IMAGE_DIR = Path('./data/raw/ai03-level1-project/test_images')
today_str = datetime.date.today().strftime("%y%m%d")

# 최종적으로 저장할 앙상블된 submission 파일 경로
ENSEMBLED_SUBMISSION_FILE = "./output/submission_ensemble_" + today_str + ".csv"

# WBF 하이퍼파라미터 (조절 가능)
IOU_THR = 0.55  # IoU가 이 값 이상인 박스들을 같은 그룹으로 묶음
SKIP_BOX_THR = 0.0001 # 이 점수 미만의 박스는 무시
WEIGHTS = [2, 1] # 각 submission 파일에 대한 가중치 (모델 성능에 따라 조절)


def main():
    print("--- Ensemble Start ---")
    
    # --- 2. 데이터 준비 ---
    # 여러 submission 파일을 하나로 합치기
    subs = [pd.read_csv(file) for file in SUBMISSION_FILES]
    combined_df = pd.concat(subs, ignore_index=True)
    
    # 이미지 ID를 기준으로 그룹화할 준비
    image_ids = combined_df['image_id'].unique()
    
    final_results = []
    
    # --- 3. WBF 핵심 로직 ---
    for image_id in tqdm(image_ids, desc="Ensembling predictions"):
        # 현재 이미지에 해당하는 모든 예측들을 가져옴
        image_df = combined_df[combined_df['image_id'] == image_id]
        
        # 원본 이미지 크기 정보 가져오기 (파일 이름 찾기)
        # image_id가 파일 이름의 숫자 부분이라고 가정
        # 이 부분은 image_id 형식에 따라 수정이 필요할 수 있음!
        image_filename = f"{image_id}.jpg" # 예: 123 -> '123.jpg' / 실제 파일이름 규칙에 맞게 수정!
        image_path = TEST_IMAGE_DIR / image_filename
        
        if not image_path.exists():
            print(f"Warning: Image file not found for image_id {image_id}. Skipping.")
            continue
            
        original_image = cv2.imread(str(image_path))
        h, w, _ = original_image.shape
        
        # WBF에 입력으로 넣기 위한 리스트 준비
        boxes_list, scores_list, labels_list = [], [], []
        
        for i in range(len(SUBMISSION_FILES)):
            model_df = subs[i]
            model_image_df = model_df[model_df['image_id'] == image_id]
            
            if model_image_df.empty:
                boxes_list.append([])
                scores_list.append([])
                labels_list.append([])
                continue

            # 좌표 변환 및 정규화: [x,y,w,h] -> [x_min, y_min, x_max, y_max] (0~1 스케일)
            boxes = model_image_df[['bbox_x', 'bbox_y', 'bbox_w', 'bbox_h']].values
            boxes[:, 2] += boxes[:, 0] # x_max
            boxes[:, 3] += boxes[:, 1] # y_max
            
            # 정규화
            boxes[:, [0, 2]] /= w
            boxes[:, [1, 3]] /= h
            
            boxes_list.append(boxes.tolist())
            scores_list.append(model_image_df['score'].values.tolist())
            labels_list.append(model_image_df['category_id'].values.tolist())
            
        # WBF 실행
        fused_boxes, fused_scores, fused_labels = weighted_boxes_fusion(
            boxes_list,
            scores_list,
            labels_list,
            weights=WEIGHTS,
            iou_thr=IOU_THR,
            skip_box_thr=SKIP_BOX_THR,
        )
        
        # 앙상블된 결과를 최종 리스트에 추가
        for box, score, label in zip(fused_boxes, fused_scores, fused_labels):
            # 좌표 역정규화 및 형식 변환
            x_min, y_min, x_max, y_max = box
            
            bbox_x = x_min * w
            bbox_y = y_min * h
            bbox_w = (x_max - x_min) * w
            bbox_h = (y_max - y_min) * h
            
            final_results.append({
                'image_id': image_id,
                'category_id': int(label),
                'bbox_x': bbox_x, 'bbox_y': bbox_y,
                'bbox_w': bbox_w, 'bbox_h': bbox_h,
                'score': score
            })

    # --- 4. 최종 제출 파일 생성 ---
    final_df = pd.DataFrame(final_results)
    final_df['annotation_id'] = range(1, len(final_df) + 1)
    final_df = final_df[['annotation_id', 'image_id', 'category_id', 'bbox_x', 'bbox_y', 'bbox_w', 'bbox_h', 'score']]
    
    final_df.to_csv(ENSEMBLED_SUBMISSION_FILE, index=False)
    
    print(f"\n--- Ensemble Finished! ---")
    print(f"Ensembled submission file saved to '{ENSEMBLED_SUBMISSION_FILE}'")
    print(final_df.head())


if __name__ == '__main__':
    main()