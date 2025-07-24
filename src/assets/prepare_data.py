# ====================================================================
# prepare_data.py
# ====================================================================
# raw 데이터 -> processed 데이터 처리하는 모듈
# 25.07.24 YOLO 전처리를 위한 yaml 모듈 추가

import json
from pathlib import Path
from typing import Dict

import pandas as pd
import utils
import yaml  # YOLO 설정을 위한 yaml 라이브러리 import

# 1. 전역 변수 및 경로 설정
BASE_DIR = Path(__file__).resolve().parent.parent.parent
PROCESSED_DATA_DIR = BASE_DIR / "data/processed"
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
RAW_DATA_DIR = BASE_DIR / "data/raw/ai03-level1-project"

TRAIN_ANNO_DIR = RAW_DATA_DIR / "train_annotations"

def expand_bbox_column(df: pd.DataFrame) -> pd.DataFrame:
    """'bbox' 리스트 컬럼을 4개의 개별 컬럼으로 확장합니다."""
    bbox_df = pd.DataFrame(
        df["bbox"].tolist(), columns=["bbox_x", "bbox_y", "bbox_w", "bbox_h"]
    )
    return pd.concat([df.drop("bbox", axis=1), bbox_df], axis=1)


def create_label_map(df: pd.DataFrame) -> Dict:
    """전체 데이터프레임 기준으로 모든 클래스를 포함하는 라벨맵을 생성합니다."""
    unique_categories = (
        df[["category_id", "class_name"]].drop_duplicates().sort_values("category_id")
    )

    id_to_idx = {
        int(row.category_id): i for i, row in enumerate(unique_categories.itertuples())
    }
    idx_to_name = {
        i: row.class_name for i, row in enumerate(unique_categories.itertuples())
    }

    return {"id_to_idx": id_to_idx, "idx_to_name": idx_to_name}


def filter_invalid_bboxes(df: pd.DataFrame) -> pd.DataFrame:
    """이미지 크기를 벗어나는 잘못된 바운딩 박스를 제거합니다."""
    invalid_x = df["bbox_x"] + df["bbox_w"] > df["width"]
    invalid_y = df["bbox_y"] + df["bbox_h"] > df["height"]
    invalid_mask = invalid_x | invalid_y

    if invalid_mask.any():
        print(
            f"--- {invalid_mask.sum()}개의 잘못된 바운딩 박스 데이터를 찾았습니다. ---"
        )
        print(
            df.loc[
                invalid_mask,
                [
                    "file_name",
                    "width",
                    "height",
                    "bbox_x",
                    "bbox_y",
                    "bbox_w",
                    "bbox_h",
                ],
            ]
        )

        filtered_df = df[
            ~invalid_mask
        ].copy()  # .copy()를 사용하여 SettingWithCopyWarning 방지
        print(
            f"\n잘못된 데이터를 제거하고, {len(filtered_df)}개의 데이터만 사용합니다."
        )
        return filtered_df
    return df


def save_artifacts(df: pd.DataFrame, label_map: Dict):
    """전처리된 데이터프레임(CSV), 라벨맵(JSON), YOLO 설정(YAML)을 저장합니다."""
    # 1. 최종 DataFrame 저장
    save_path = PROCESSED_DATA_DIR / "train_df.csv"
    df.to_csv(save_path, index=False)
    print(f"\n전처리된 데이터를 '{save_path}'에 저장했습니다.")

    # 2. 라벨맵 JSON 저장
    label_map_path = PROCESSED_DATA_DIR / "label_map.json"
    with open(label_map_path, "w", encoding="utf-8") as f:
        json.dump(label_map, f, ensure_ascii=False, indent=4)
    print(f"라벨 매핑 정보를 '{label_map_path}'에 저장했습니다.")

    # 3. YOLO 학습용 data.yaml 파일 저장
    yolo_data_path = (
        PROCESSED_DATA_DIR / "YOLO_data.yaml"
    )  # 프로젝트 최상위 경로에 저장

    # train/val 이미지가 저장된 경로 (실제 프로젝트 구조에 맞게 수정 필요)
    # 여기서는 모든 이미지가 train_images 폴더에 있다고 가정
    train_img_path = str(RAW_DATA_DIR / "train_images")

    yolo_config = {
        "path": str(RAW_DATA_DIR),  # 데이터셋 루트 경로
        "train": train_img_path,  # train 이미지 경로
        "val": train_img_path,  # val 이미지 경로 (동일하게 설정)
        "names": label_map["idx_to_name"],  # 라벨맵에서 클래스 이름 가져오기
    }

    with open(yolo_data_path, "w") as f:
        yaml.dump(yolo_config, f, default_flow_style=False, sort_keys=False)
    print(f"YOLO 설정 파일을 '{yolo_data_path}'에 저장했습니다.")


def main():
    """데이터 전처리 파이프라인을 실행하는 메인 함수"""
    print("--- 데이터 전처리 시작 ---")

    # 1. 데이터 로드 및 기본 가공
    # train_df = load_dummy_data()
    train_df = utils.parse_raw_annotations(TRAIN_ANNO_DIR)
    train_df = expand_bbox_column(train_df)

    # 2. 전체 데이터를 기준으로 라벨맵 생성
    label_map = create_label_map(train_df)

    # 3. 잘못된 데이터 필터링
    train_df = filter_invalid_bboxes(train_df)

    # 4. 생성해둔 라벨맵을 사용해 라벨 인덱스 적용
    train_df["label_idx"] = train_df["category_id"].map(label_map["id_to_idx"])

    # 5. 결과물 저장
    save_artifacts(train_df, label_map)

    print("\n--- 데이터 전처리 및 저장 완료! ---")
    print("최종 데이터프레임:")
    print(train_df.head())
    print("\n생성된 라벨맵:")
    print(label_map)


if __name__ == "__main__":
    main()
