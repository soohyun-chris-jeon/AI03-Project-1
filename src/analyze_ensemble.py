# src/analyze_ensemble.py (리팩토링 최종 버전)

import json
import os
import re
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

# --- 1. 설정: 분석할 파일 및 경로 ---
SUBMISSION_FILES = {
    "YOLO": Path("./output/submission_YOLO.csv"),
    "ResNet": Path("./output/submission_250729(2)_RESNET.csv"),
    "Ensemble": Path("./output/submission_ensemble_250730(2).csv"),
}
TEST_IMAGE_DIR = Path("./data/raw/ai03-level1-project/test_images")
LABEL_MAP_PATH = Path("./data/processed/label_map.json")

# 결과물을 저장할 경로
OUTPUT_ANALYSIS_DIR = Path("./output/analysis")
OUTPUT_VISUALIZATION_DIR = Path("./output/figures/ensemble_misses_visualization")

# 시각화할 이미지 개수 (각 케이스별 상위 N개)
NUM_VISUALIZE = 100

# 모델별 색상 (RGB 순서)
COLORS = {
    "YOLO": (0, 0, 255),  # Google Blue
    "ResNet": (52, 168, 83),  # Google Green
    "Ensemble": (255, 0, 0),  # Google Red
}

# --- 폰트 설정 (사용자 환경에 맞게 경로 수정 필수!) ---
# Windows: font_path = "C:/Windows/Fonts/malgun.ttf"
# macOS:   font_path = "/System/Library/Fonts/Supplemental/AppleGothic.ttf"
# Linux (나눔폰트 설치 시): font_path = "/usr/share/fonts/truetype/nanum/NanumGothic.ttf"
FONT_PATH = "/usr/share/fonts/truetype/nanum/NanumGothic.ttf"


# --- 2. 헬퍼 함수 ---
def get_font(font_path: str, size: int = 15) -> ImageFont.FreeTypeFont:
    """폰트 파일을 로드하고, 없을 경우 기본 폰트로 대체합니다."""
    try:
        return ImageFont.truetype(font_path, size)
    except IOError:
        print(f"Warning: Font not found at '{font_path}'. Using default font.")
        return ImageFont.load_default()


def extract_image_id(filename: str) -> int:
    """파일 이름에서 숫자 ID를 추출합니다."""
    try:
        return int("".join(re.findall(r"\d+", filename)))
    except (ValueError, TypeError):
        return hash(filename)


def draw_predictions_on_image(
    image_path: Path,
    dfs: dict,
    image_id: int,
    id_to_name_map: dict,
    font: ImageFont.FreeTypeFont,
):
    """Pillow를 사용해 3개 모델의 예측을 하나의 이미지에 모두 그립니다."""
    try:
        pil_image = Image.open(image_path).convert("RGB")
        draw = ImageDraw.Draw(pil_image)
    except FileNotFoundError:
        print(f"  - Error: Could not open image at {image_path}")
        return None

    for model_name, color in COLORS.items():
        predictions = dfs[model_name][dfs[model_name]["image_id"] == image_id]
        for _, row in predictions.iterrows():
            x, y, w, h = row["bbox_x"], row["bbox_y"], row["bbox_w"], row["bbox_h"]
            xmin, ymin, xmax, ymax = int(x), int(y), int(x + w), int(y + h)

            draw.rectangle([xmin, ymin, xmax, ymax], outline=color, width=3)

            score = row["score"]
            category_id = row["category_id"]
            class_name = id_to_name_map.get(str(category_id), "Unknown")
            label_text = f"{model_name[:3]}: {class_name} ({score:.2f})"

            text_bbox = draw.textbbox((xmin, ymin - 20), label_text, font=font)
            draw.rectangle(text_bbox, fill=color)
            draw.text((xmin, ymin - 20), label_text, font=font, fill=(255, 255, 255))

    return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)


# --- 3. 메인 분석 함수 ---
def analyze_and_visualize():
    """데이터 로드, 앙상블 성능 분석, 특정 케이스 시각화를 수행하는 메인 함수"""
    print("--- Starting Ensemble Analysis and Visualization ---")

    OUTPUT_ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_VISUALIZATION_DIR.mkdir(parents=True, exist_ok=True)
    font = get_font(FONT_PATH)

    try:
        dfs = {name: pd.read_csv(path) for name, path in SUBMISSION_FILES.items()}
        with open(LABEL_MAP_PATH, "r", encoding="utf-8") as f:
            label_map = json.load(f)
        id_to_name_map = label_map["id_to_name"]
        test_files = os.listdir(TEST_IMAGE_DIR)
        id_to_filename_map = {extract_image_id(f): f for f in test_files}
        print("All data files loaded successfully.")
    except FileNotFoundError as e:
        print(f"FATAL ERROR: Could not load a required file. {e}")
        return

    for df in dfs.values():
        df["detection_key"] = (
            df["image_id"].astype(str) + "_" + df["category_id"].astype(str)
        )

    yolo_keys = set(dfs["YOLO"]["detection_key"])
    resnet_keys = set(dfs["ResNet"]["detection_key"])
    ensemble_keys = set(dfs["Ensemble"]["detection_key"])

    analysis_cases = {
        "ensemble_vs_yolo": ensemble_keys - yolo_keys,
        "ensemble_vs_resnet": ensemble_keys - resnet_keys,
        "ensemble_vs_both": (ensemble_keys - yolo_keys).intersection(
            ensemble_keys - resnet_keys
        ),
    }

    for case_name, missed_keys in analysis_cases.items():
        if not missed_keys:
            print(f"\nNo cases found for '{case_name}'.")
            continue

        missed_df = dfs["Ensemble"][
            dfs["Ensemble"]["detection_key"].isin(missed_keys)
        ].copy()
        missed_df.drop(columns=["detection_key"], inplace=True)

        output_csv_path = OUTPUT_ANALYSIS_DIR / f"analysis_{case_name}.csv"
        missed_df.to_csv(output_csv_path, index=False)

        print(f"\n--- Analysis for '{case_name}' ({len(missed_df)} cases) ---")
        print(f"Result saved to: {output_csv_path}")
        print(missed_df.head())

        print(f"Visualizing top {NUM_VISUALIZE} cases for '{case_name}'...")
        image_ids_to_draw = missed_df["image_id"].unique()[:NUM_VISUALIZE]

        for image_id in image_ids_to_draw:
            filename = id_to_filename_map.get(image_id)
            if not filename:
                print(f"  - Warning: Filename for image_id {image_id} not found.")
                continue

            image_path = TEST_IMAGE_DIR / filename
            vis_image = draw_predictions_on_image(
                image_path, dfs, image_id, id_to_name_map, font
            )

            if vis_image is not None:
                output_vis_path = (
                    OUTPUT_VISUALIZATION_DIR / f"{case_name}_image_{image_id}.png"
                )
                cv2.imwrite(str(output_vis_path), vis_image)
                print(f"  - Saved visualization to: {output_vis_path}")

    print("\n--- All tasks finished! ---")


# --- 4. 스크립트 실행 ---
if __name__ == "__main__":
    analyze_and_visualize()
