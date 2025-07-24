# src/data/utils.py
import json
from pathlib import Path

import pandas as pd
from tqdm import tqdm


def collate_fn(batch):
    """
    DataLoader가 배치(batch)를 구성할 때 호출되는 함수.
    이미지와 타겟을 각각의 튜플로 묶어 반환한다.
    """
    return tuple(zip(*batch))


def parse_raw_annotations(ann_dir: Path) -> pd.DataFrame:
    """3중 폴더 구조의 원본 어노테이션을 파싱하여 DataFrame으로 반환"""
    all_annotations = []
    image_level_dirs = [d for d in ann_dir.iterdir() if d.is_dir()]

    for image_dir_path in tqdm(image_level_dirs, desc="[L1] Images"):
        pill_level_dirs = [d for d in image_dir_path.iterdir() if d.is_dir()]
        for pill_dir_path in pill_level_dirs:
            json_files = list(pill_dir_path.glob("*.json"))
            if not json_files:
                continue

            json_file_path = json_files[0]
            try:
                with open(json_file_path, "r", encoding="utf-8") as f:
                    ann_data = json.load(f)
                    image_info = ann_data.get("images", [{}])[0]
                    annotation_info = ann_data.get("annotations", [{}])[0]
                    category_info = ann_data.get("categories", [{}])[0]
                    all_annotations.append(
                        {
                            "image_id": image_info.get("id"),
                            "file_name": image_info.get("file_name"),
                            "width": image_info.get("width"),
                            "height": image_info.get("height"),
                            "category_id": category_info.get("id"),
                            "class_name": category_info.get("name"),
                            "bbox": annotation_info.get("bbox"),
                        }
                    )
            except Exception as e:
                print(f"\n파일 처리 에러: {json_file_path}, 에러: {e}")

    return pd.DataFrame(all_annotations)
