# ====================================================================
# prepare_data.py
# ====================================================================
# raw 데이터 -> processed 데이터 처리하는 모듈

# src/data/prepare_data.py

import json
from pathlib import Path

import pandas as pd

# 1. 전역 변수 및 경로 설정
BASE_DIR = Path(__file__).resolve().parent.parent.parent  # src/data -> project_root
RAW_DATA_DIR = BASE_DIR / "data/raw/ai03-level1-project"
PROCESSED_DATA_DIR = BASE_DIR / "data/processed"
SAVE_PATH = PROCESSED_DATA_DIR / "train_df.csv"
TRAIN_ANNO_DIR = RAW_DATA_DIR / "train_annotations"

# parse_raw_annotations 함수는 별도로 정의되어 있다고 가정
# from .utils import parse_raw_annotations


def main():
    """
    Raw 데이터를 불러와 전처리하고, 학습용 CSV와 라벨맵 JSON을 저장하는 메인 함수
    """
    print("--- 데이터 전처리 시작 ---")
    # parse_raw_annotations 함수가 정의되어 있어야 함
    # train_df = parse_raw_annotations(TRAIN_ANNO_DIR)

    # --- (임시) 함수가 없으므로 임시 데이터프레임으로 대체 ---
    # 실제로는 parse_raw_annotations 함수를 호출해야 함
    dummy_data = {
        "file_name": ["1.jpg", "2.jpg", "3.jpg"],
        "width": [1024, 1024, 800],
        "height": [768, 768, 600],
        "bbox": [
            [10, 10, 50, 50],
            [20, 20, 1050, 60],
            [30, 30, 40, 40],
        ],  # 2번 데이터는 오류 데이터
        "category_id": [10, 12, 10],
        "class_name": ["cat", "dog", "cat"],
    }
    train_df = pd.DataFrame(dummy_data)
    # ---------------------------------------------------

    # --- 1. bbox 컬럼 분리 ---
    bbox_df = pd.DataFrame(
        train_df["bbox"].tolist(), columns=["bbox_x", "bbox_y", "bbox_w", "bbox_h"]
    )
    train_df = pd.concat([train_df.drop("bbox", axis=1), bbox_df], axis=1)

    # --- 2. 잘못된 바운딩 박스 데이터 제거 ---
    invalid_x = train_df["bbox_x"] + train_df["bbox_w"] > train_df["width"]
    invalid_y = train_df["bbox_y"] + train_df["bbox_h"] > train_df["height"]
    invalid_rows = train_df[invalid_x | invalid_y]

    if not invalid_rows.empty:
        print(
            f"--- {len(invalid_rows)}개의 잘못된 바운딩 박스 데이터를 찾았습니다. ---"
        )
        print(
            invalid_rows[
                ["file_name", "width", "height", "bbox_x", "bbox_y", "bbox_w", "bbox_h"]
            ]
        )
        train_df = train_df[~(invalid_x | invalid_y)]
        print(f"\n잘못된 데이터를 제거하고, {len(train_df)}개의 데이터만 사용합니다.")

    # --- 3. category_id를 새로운 label_idx로 매핑 ---
    unique_category_ids = sorted(train_df["category_id"].unique())
    id_to_idx = {
        int(original_id): idx for idx, original_id in enumerate(unique_category_ids)
    }  # 배경 클래스를 위해 0부터 시작
    train_df["label_idx"] = train_df["category_id"].map(id_to_idx)

    # --- 4. 결과물 저장 ---
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    # 라벨맵 저장
    label_map = {
        "id_to_idx": id_to_idx,
        "idx_to_id": {idx: int(original_id) for original_id, idx in id_to_idx.items()},
        "id_to_name": dict(zip(train_df["category_id"], train_df["class_name"])),
    }
    with open(PROCESSED_DATA_DIR / "label_map.json", "w", encoding="utf-8") as f:
        json.dump(label_map, f, ensure_ascii=False, indent=4)
    print(
        f"\n라벨 매핑 정보를 '{PROCESSED_DATA_DIR / 'label_map.json'}'에 저장했습니다."
    )

    # 최종 DataFrame 저장
    train_df.to_csv(SAVE_PATH, index=False)
    print(f"전처리된 데이터를 '{SAVE_PATH}'에 저장했습니다.")

    print("\n--- 데이터 전처리 및 저장 완료! ---")
    print(train_df.head())


if __name__ == "__main__":
    main()
