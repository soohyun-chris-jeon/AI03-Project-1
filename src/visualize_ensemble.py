# src/visualize_ensemble.py

import pandas as pd
import cv2
import os
import json
import re
from pathlib import Path

# --- 1. 설정: 분석할 파일 및 경로 ---
# (경로는 네 실제 프로젝트 구조에 맞게 확인/수정해 줘)

# 분석하고 싶은 이미지 ID 리스트 (analysis...csv 파일에서 찾은 ID들을 여기에 넣어줘)
IMAGE_IDS_TO_VISUALIZE = [914] # 예시 ID

# 3개 모델의 예측 결과 파일
SUBMISSION_FILES = {
    "YOLO": Path("./output/submission_YOLO.csv"),
    "ResNet": Path("./output/submission_250729(2)_RESNET.csv"),
    "Ensemble": Path("./output/submission_ensemble_250730(2).csv")
}

# # --- 1. 설정: 분석할 파일들의 경로를 정확하게 입력해주세요 ---
# SUBMISSION_FILES = {
#     "YOLO": "./output/submission_YOLO.csv",
#     "ResNet": "./output/submission_250729(2)_RESNET.csv",
#     # "MobileNet": "./output/submission_250730_mobilenet.csv",
#     "Ensemble": "./output/submission_ensemble_250730(2).csv",
#     # "ResNet05": "./output/submission_250730_RESNET_FINAL.csv",
# }


# 원본 테스트 이미지 폴더
TEST_IMAGE_DIR = Path("./data/raw/ai03-level1-project/test_images")

# 라벨 맵 파일 (category_id -> 클래스 이름 변환용)
LABEL_MAP_PATH = Path("./data/processed/label_map.json")

# 결과 이미지를 저장할 폴더
OUTPUT_DIR = Path("./output/figures/ensemble_visualizations")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# --- 2. 헬퍼 함수 ---
def extract_image_id(filename: str) -> int:
    """파일 이름에서 숫자 ID를 추출하는 함수"""
    try:
        return int("".join(re.findall(r'\d+', filename)))
    except:
        return hash(filename)

def draw_boxes(image, df, image_id, model_name, color, id_to_name_map):
    """특정 모델의 예측 박스를 이미지에 그리는 함수"""
    predictions = df[df['image_id'] == image_id]
    
    for _, row in predictions.iterrows():
        x, y, w, h = row['bbox_x'], row['bbox_y'], row['bbox_w'], row['bbox_h']
        xmin, ymin = int(x), int(y)
        xmax, ymax = int(x + w), int(y + h)
        
        score = row['score']
        category_id = row['category_id']
        class_name = id_to_name_map.get(str(category_id), 'Unknown') # json key는 문자열
        
        # 박스 그리기
        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color, 2)
        
        # 라벨 텍스트 만들기
        label_text = f"{model_name}: {class_name[:10]} ({score:.2f})"
        
        # 텍스트 배경 및 텍스트 그리기
        (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        cv2.rectangle(image, (xmin, ymin - 20), (xmin + tw, ymin), color, -1)
        cv2.putText(image, label_text, (xmin, ymin - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
    return image


# --- 3. 메인 실행 로직 ---
if __name__ == '__main__':
    # 데이터 로드
    dfs = {name: pd.read_csv(path) for name, path in SUBMISSION_FILES.items()}
    with open(LABEL_MAP_PATH, 'r', encoding='utf-8') as f:
        label_map = json.load(f)
    id_to_name_map = label_map['id_to_name']

    # image_id와 실제 파일 이름 매핑 생성
    test_files = os.listdir(TEST_IMAGE_DIR)
    id_to_filename_map = {extract_image_id(f): f for f in test_files}
    
    # 모델별 색상 지정
    COLORS = {
        "YOLO": (255, 0, 0),      # 파란색
        "ResNet": (0, 255, 0),    # 초록색
        "Ensemble": (0, 0, 255)   # 빨간색
    }
    
    print("--- Starting Visualization ---")
    for image_id in IMAGE_IDS_TO_VISUALIZE:
        filename = id_to_filename_map.get(image_id)
        if not filename:
            print(f"Warning: Filename for image_id {image_id} not found. Skipping.")
            continue
        
        image_path = TEST_IMAGE_DIR / filename
        image = cv2.imread(str(image_path))
        
        # 각 모델의 예측을 순서대로 이미지에 그리기
        image = draw_boxes(image, dfs['YOLO'], image_id, "YOLO", COLORS['YOLO'], id_to_name_map)
        image = draw_boxes(image, dfs['ResNet'], image_id, "ResNet", COLORS['ResNet'], id_to_name_map)
        image = draw_boxes(image, dfs['Ensemble'], image_id, "Ensemble", COLORS['Ensemble'], id_to_name_map)
        
        # 결과 이미지 저장
        output_path = OUTPUT_DIR / f"visualization_{image_id}.png"
        cv2.imwrite(str(output_path), image)
        print(f"Visualization for image_id {image_id} saved to '{output_path}'")
        
    print("\n--- Visualization Finished! ---")