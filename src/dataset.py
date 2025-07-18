# ====================================================================
# dataset.py
# ====================================================================
# 이미지를 불러오고, 전처리하고, 모델에 배치(batch) 단위로 전달하는 역할을 함

from pathlib import Path  # payhon path

# Data Augmentation 패키지: Albumentations
import albumentations as A
import cv2  # OpenCV - 고급 이미지/비디오 처리
import torch
from torch.utils.data import Dataset  # 커스텀 데이터셋, 배치 로딩

# PyTorch 이미지 전처리
# PyTorch 데이터 처리


# 데이터 증강 (Augmentation) : Albumentations 라이브러리 사용
# 를 사용하는 것이 바운딩 박스 변환에 더 유리
train_transforms = A.Compose(
    [
        A.Resize(512, 512),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.2),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        # PyTorch 텐서로 변환
        A.pytorch.ToTensorV2(),
    ],
    bbox_params=A.BboxParams(format="pascal_voc", label_fields=["labels"]),
)  # bbox 형식은 pascal_voc: [xmin, ymin, xmax, ymax]

test_transforms = A.Compose(
    [
        A.Resize(512, 512),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        A.pytorch.ToTensorV2(),
    ],
    bbox_params=A.BboxParams(format="pascal_voc", label_fields=["labels"]),
)


# DataLoader를 위한 collate_fn. 이미지와 타겟을 리스트로 묶어줌
def collate_fn(batch):
    return tuple(zip(*batch))


# PillDataset 데이터셋 클래스
class PillDataset(Dataset):
    # --- mode 파라미터 추가 및 df를 직접 받도록 수정 ---
    def __init__(self, df, image_dir, mode="train", transforms=None):
        self.df = df
        self.image_dir = Path(image_dir)
        self.mode = mode
        self.transforms = transforms

        # --- image_ids를 미리 뽑아 중복을 제거 ---
        # df['file_name']을 사용하면 이미지 파일 이름으로 고유한 이미지를 식별 가능.
        self.image_ids = self.df["file_name"].unique()

    def __len__(self):
        # --- 고유한 이미지의 개수를 반환 ---
        return len(self.image_ids)

    def __getitem__(self, idx):
        # 이미지 ID(파일 이름)를 가져옴
        image_id = self.image_ids[idx]
        image_path = self.image_dir / image_id

        # 이미지 로드
        image = cv2.imread(str(image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # --- mode에 따라 다르게 동작 ---
        # 학습 모드일 경우, 어노테이션(target)까지 준비
        if self.mode == "train":
            # 현재 이미지에 해당하는 모든 어노테이션(알약)을 df에서 필터링
            records = self.df[self.df["file_name"] == image_id]

            boxes = records[["bbox_x", "bbox_y", "bbox_w", "bbox_h"]].values
            # bbox: [x, y, w, h] -> [xmin, ymin, xmax, ymax]
            boxes[:, 2] = boxes[:, 0] + boxes[:, 2]
            boxes[:, 3] = boxes[:, 1] + boxes[:, 3]

            labels = records["category_id"].values

            target = {
                "boxes": torch.as_tensor(boxes, dtype=torch.float32),
                "labels": torch.as_tensor(labels, dtype=torch.int64),
            }

            # Augmentation 적용
            if self.transforms:
                transformed = self.transforms(
                    image=image, bboxes=target["boxes"], labels=target["labels"]
                )
                image = transformed["image"]
                target["boxes"] = torch.as_tensor(
                    transformed["bboxes"], dtype=torch.float32
                )
                if len(target["boxes"]) == 0:
                    target["boxes"] = torch.zeros((0, 4), dtype=torch.float32)

            return image, target

        # 테스트 모드일 경우, 이미지와 파일 이름만 반환
        elif self.mode == "test":
            # 테스트 시에는 보통 기본적인 리사이즈, 정규화만 적용
            if self.transforms:
                transformed = self.transforms(image=image)
                image = transformed["image"]

            # 나중에 예측 결과를 이미지와 매칭시키기 위해 파일 이름을 반환
            return image, image_id
