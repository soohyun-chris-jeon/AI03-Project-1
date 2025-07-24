# ====================================================================
# train.py
# ====================================================================
# 공통으로 사용될 함수들을 모아두는 곳
# src/utils.py

import json
import os
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm


def seed_everything(seed: int):
    """
    재현성을 위해 모든 난수 생성기의 시드를 고정하는 함수.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"Seed set to {seed}")


def collate_fn(batch):
    """
    Object Detection DataLoader를 위한 collate_fn.
    이미지와 타겟을 리스트로 묶어줍니다.
    """
    return tuple(zip(*batch))


def visualize_sample(dataset, classes):
    idx = random.randint(0, len(dataset) - 1)
    image_tensor, target = dataset[idx]

    # 텐서를 시각화를 위해 NumPy 배열로 변환
    image = image_tensor.permute(1, 2, 0).numpy()

    # 역정규화 (선택 사항, 이미지가 이상하게 보일 때)
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = (image * std + mean).clip(0, 1)

    boxes = target["boxes"].numpy()
    labels = target["labels"].numpy()

    plt.figure(figsize=(8, 8))
    plt.imshow(image)

    for box, label_idx in zip(boxes, labels):
        xmin, ymin, xmax, ymax = box
        class_name = classes[label_idx]

        # 바운딩 박스 그리기
        plt.gca().add_patch(
            plt.Rectangle(
                (xmin, ymin),
                xmax - xmin,
                ymax - ymin,
                edgecolor="red",
                facecolor="none",
                lw=2,
            )
        )
        # 라벨 텍스트 추가
        plt.text(xmin, ymin, class_name, bbox=dict(facecolor="yellow", alpha=0.5))

    plt.axis("off")
    plt.show()


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


class EarlyStopping:
    """주어진 patience 후 검증 점수가 향상되지 않으면 학습을 조기 중단시킵니다."""

    def __init__(
        self,
        patience=7,
        verbose=False,
        delta=0,
        mode="min",
        path="./experiments/best_model.pt",
        evaluation_name="score",
    ):
        """
        Args:
            patience (int): 검증 점수가 향상된 후 기다릴 에폭 수.
            verbose (bool): True일 경우, 점수가 향상될 때마다 메시지를 출력.
            delta (float): 점수 향상으로 인정될 최소 변화량.
            mode (str): 'min' 또는 'max'. min은 점수가 낮아지는 것을, max는 높아지는 것을 목표로 함.
            path (str): 모델 체크포인트 저장 경로.
            evaluation_name (str): 로그에 표시될 평가 지표의 이름.
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.delta = delta
        self.path = path
        self.mode = mode
        self.evaluation_name = evaluation_name

        if self.mode == "min":
            self.val_score = np.inf
        else:
            self.val_score = -np.inf

    def __call__(self, score, model):

        # mode에 따른 점수 비교
        is_best = False
        if self.mode == "min":
            if (
                score < self.best_score - self.delta
                if self.best_score is not None
                else True
            ):
                is_best = True
        else:  # mode == 'max'
            if (
                score > self.best_score + self.delta
                if self.best_score is not None
                else True
            ):
                is_best = True

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(score, model)
        elif is_best:
            self.best_score = score
            self.save_checkpoint(score, model)
            self.counter = 0
        else:
            self.counter += 1
            print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True

    def save_checkpoint(self, score, model):
        """Saves model when validation score improves."""
        if self.verbose:
            # mode에 따라 'decreased' 또는 'increased'를 동적으로 표현
            change_text = "decreased" if self.mode == "min" else "increased"
            print(
                f"{self.evaluation_name} {change_text} ({self.val_score:.6f} --> {score:.6f}). Saving model ..."
            )

        save_dir = os.path.dirname(self.path)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            print(f"Created directory: {save_dir}")

        torch.save(model.state_dict(), self.path)
        self.val_score = score


def get_unique_filepath(filepath: Path) -> Path:
    """
    주어진 파일 경로가 이미 존재할 경우, 파일 이름 뒤에 (2), (3)... 등을 붙여
    중복되지 않는 새로운 경로를 반환합니다.
    """
    if not filepath.exists():
        return filepath

    parent = filepath.parent
    stem = filepath.stem  # 파일 이름 (확장자 제외)
    suffix = filepath.suffix  # 확장자

    counter = 2
    while True:
        new_filepath = parent / f"{stem}({counter}){suffix}"
        if not new_filepath.exists():
            return new_filepath
        counter += 1
