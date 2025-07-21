# ====================================================================
# train.py
# ====================================================================
# 공통으로 사용될 함수들을 모아두는 곳
# src/utils.py

import os
import random

import matplotlib.pyplot as plt
import numpy as np
import torch


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


# 사용 예시 (train_loader 생성 후)
print("--- 증강이 적용된 훈련 데이터 샘플 ---")
visualize_sample(train_dataset, classes)

print("--- 원본 검증 데이터 샘플 ---")
visualize_sample(val_dataset, classes)
