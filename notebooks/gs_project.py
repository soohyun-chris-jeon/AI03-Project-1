import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.transforms import functional as F
from torch.utils.data import DataLoader, Dataset # PyTorch의 Dataset 클래스를 상속받기 위한 모듈
import matplotlib.patches as patches
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.pyplot as plt
import os              # 파일 및 디렉토리 경로를 다루기 위한 표준 라이브러리
import json            # JSON 파일을 읽고 쓰기 위한 표준 라이브러리
import torch           # 딥러닝 라이브러리 PyTorch (텐서 연산 등)
from PIL import Image  # 이미지를 다루기 위한 Pillow 라이브러리
from torchvision import transforms
from pathlib import Path
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torch.optim import Adam
import torch.nn as nn
import matplotlib
from pathlib import Path  # payhon path
import albumentations as A
import cv2  # OpenCV - 고급 이미지/비디오 처리
import pandas as pd
from tqdm import tqdm
from torchmetrics.detection.mean_ap import MeanAveragePrecision
# Garbage Collector 모듈
import gc


# (1) 시스템별 폰트
import platform
if platform.system() == 'Darwin':  # Mac
    font_name = 'AppleGothic'

matplotlib.rc('font', family=font_name)
# 마이너스 기호 깨짐 방지
plt.rcParams['axes.unicode_minus'] = False


# 메모리 정리 루틴
gc.collect()
torch.cuda.empty_cache()


# --- 디바이스 설정 ---
DEVICE = torch.device("cpu") if torch.backends.mps.is_available() else torch.device("cpu")
print(f"사용 중인 디바이스: {DEVICE}")
SEED = 42



# --- 학습 하이퍼파라미터 ---
BATCH_SIZE = 16
NUM_EPOCHS = 10
LEARNING_RATE = 1e-4
MOMENTUM = 0.9
WEIGHT_DECAY = 0.0005

# --- 데이터 경로 설정 ---
DATA_ROOT = "/Users/jogyeseung/Desktop/AI03-Project-1/data/raw"  # 본인 데이터의 절대경로
TRAIN_IMAGE_DIR = os.path.join(DATA_ROOT, "train_images")
TRAIN_ANNO_DIR = os.path.join(DATA_ROOT, "train_annotations")
TEST_IMAGE_DIR = os.path.join(DATA_ROOT, "test_images")
# PROCESSED_TRAIN_CSV = "../data/processed/train_df.csv"  # 데이터 전처리된 csv 파일

# --- 모델 설정 ---
NUM_CLASSES = 73
MODEL_NAME = "fasterrcnn_resnet50_fpn"
USE_PRETRAINED = True

# --- 학습 고도화 설정 ---
USE_SCHEDULER = True  # Learning rate scheduler 사용 여부
EARLY_STOPPING = True  # Early stopping 적용 여부
AUGMENTATION = True  # 데이터 증강 사용 여부

# --- 실험 로깅용 설정 ---
USE_WANDB = True
WANDB_PROJECT = "AI03-Project-1"
RUN_NAME = f"{MODEL_NAME}_bs{BATCH_SIZE}_lr{LEARNING_RATE}"


# --- 실험 결과 저장 경로 ---
EXPERIMENT_DIR = "/Users/jogyeseung/Desktop/AI03-Project-1/experiments"



def parse_raw_annotations(ann_dir: Path) -> pd.DataFrame:
    """
    복잡한 3중 폴더 구조의 원본 어노테이션을 파싱하여
    하나의 Pandas DataFrame으로 반환하는 함수.
    """
    all_annotations = []

    # Level 1: 이미지별 폴더 순회
    image_level_dirs = os.listdir(ann_dir)
    for image_dir_name in tqdm(image_level_dirs, desc="[L1] Images"):
        image_dir_path = ann_dir / image_dir_name
        if not image_dir_path.is_dir():
            continue

        # Level 2: 알약 종류 폴더 순회
        pill_level_dirs = os.listdir(image_dir_path)
        for pill_dir_name in pill_level_dirs:
            pill_dir_path = image_dir_path / pill_dir_name
            if not pill_dir_path.is_dir():
                continue

            # Level 3: 실제 .json 파일 파싱
            json_files = [f for f in os.listdir(pill_dir_path) if f.endswith(".json")]
            if not json_files:
                continue

            # 첫 번째 json 파일만 사용
            json_file_path = pill_dir_path / json_files[0]

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

# 1. 핵심 함수를 호출해서 DataFrame 생성
from pathlib import Path

# --- 데이터 경로 설정 ---
DATA_ROOT = "/Users/jogyeseung/Desktop/AI03-Project-1/data/raw"
TRAIN_IMAGE_DIR = Path(DATA_ROOT) / "train_images"
TRAIN_ANNO_DIR = Path(DATA_ROOT) / "train_annotations"
TEST_IMAGE_DIR = Path(DATA_ROOT) / "test_images"

# --- 데이터 전처리된 csv 파일 저장 경로 ---
PROCESSED_DATA_DIR = Path("/Users/jogyeseung/Desktop/AI03-Project-1/data/processed")
SAVE_PATH = PROCESSED_DATA_DIR / "train_df.csv"
# --- 핵심 함수 호출해서 DataFrame 생성 ---
train_df = parse_raw_annotations(TRAIN_ANNO_DIR)


# --- (1). bbox 컬럼을 4개로 분리 ---
# bbox 컬럼 분리
bbox_df = pd.DataFrame(
    train_df["bbox"].tolist(), columns=["bbox_x", "bbox_y", "bbox_w", "bbox_h"]
)
train_df = pd.concat([train_df.drop("bbox", axis=1), bbox_df], axis=1)

# ✨ --- [핵심 수정] 잘못된 바운딩 박스 데이터 제거 ---
# xmax (bbox_x + bbox_w)가 이미지 너비(width)를 초과하는 경우
invalid_x = train_df["bbox_x"] + train_df["bbox_w"] > train_df["width"]
# ymax (bbox_y + bbox_h)가 이미지 높이(height)를 초과하는 경우
invalid_y = train_df["bbox_y"] + train_df["bbox_h"] > train_df["height"]

# 잘못된 데이터를 필터링
invalid_rows = train_df[invalid_x | invalid_y]
if not invalid_rows.empty:
    print(f"--- {len(invalid_rows)}개의 잘못된 바운딩 박스 데이터를 찾았습니다. ---")
    print(
        invalid_rows[
            ["file_name", "width", "height", "bbox_x", "bbox_y", "bbox_w", "bbox_h"]
        ]
    )

    # 유효한 데이터만 남김
    train_df = train_df[~(invalid_x | invalid_y)]
    print(f"\n잘못된 데이터를 제거하고, {len(train_df)}개의 데이터만 사용합니다.")

# --- (2). category_id를 새로운 label_idx로 매핑 ---
# 고유한 category_id 목록을 뽑아 정렬
unique_category_ids = sorted(train_df["category_id"].unique())
NUM_CLASSES = len(unique_category_ids)
# category_id를 0, 1, 2... 인덱스로 변환하는 딕셔너리 생성
id_to_idx = {
    int(original_id): idx
    for idx, original_id in enumerate(
        unique_category_ids, start=1
    )  # <--- start=1 추가!
}
# 이 매핑 정보를 사용해서 'label_idx'라는 새 컬럼을 추가
train_df["label_idx"] = train_df["category_id"].map(id_to_idx)

# 나중에 추론 결과에서 원래 클래스 이름을 찾을 수 있도록 매핑 정보도 저장
label_map = {
    "id_to_idx": id_to_idx,
    "idx_to_id": {idx: int(original_id) for original_id, idx in id_to_idx.items()},
    "id_to_name": dict(zip(train_df["category_id"], train_df["class_name"])),
}
with open(PROCESSED_DATA_DIR / "label_map.json", "w", encoding="utf-8") as f:
    json.dump(label_map, f, ensure_ascii=False, indent=4)

print(f"\n총 {len(unique_category_ids)}개의 고유 클래스를 발견했습니다.")
print("라벨 매핑 정보를 'data/processed/label_map.json'에 저장했습니다.")


# 3. 최종 DataFrame을 CSV 파일로 저장
train_df.to_csv(SAVE_PATH, index=False)

print(f"\n--- 데이터 전처리 및 저장 완료! ---")
print(train_df.head())

print(train_df["label_idx"].value_counts().plot(kind="bar"))

print(train_df["class_name"].value_counts())

classes_tmp = train_df["class_name"].unique()
print(classes_tmp)

# 1. 'category_id'와 'class_name' 컬럼으로 고유한 쌍을 찾고, ID 기준으로 정렬
class_mapping_df = (
    train_df[["category_id", "class_name"]]
    .drop_duplicates()
    .sort_values(by="category_id")
)
# 2. 정렬된 DataFrame에서 클래스 이름만 리스트로 추출
sorted_class_names = [
    name.split("(")[0].strip() for name in class_mapping_df["class_name"]
]
# 3. 맨 앞에 'background' 추가
classes = ["background"] + sorted_class_names

print(f"총 클래스 개수 (배경 포함): {len(classes)}")
print(f"클래스 리스트: {classes}")

# Load a model
# (1) 데이터 증강 (Augmentation) : Albumentations 라이브러리 사용
train_transforms = A.Compose(
    [
        A.Resize(512, 512),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.2),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        # PyTorch 텐서로 변환
        A.pytorch.ToTensorV2(),
    ],
    bbox_params=A.BboxParams(format="albumentations", label_fields=["labels"]),
)  # bbox 형식은 pascal_voc: [xmin, ymin, xmax, ymax]

val_transforms = A.Compose(
    [
        A.Resize(512, 512),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        A.pytorch.ToTensorV2(),
    ],
    bbox_params=A.BboxParams(format="albumentations", label_fields=["labels"]),
)

test_transforms = A.Compose(
    [
        A.Resize(512, 512),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        A.pytorch.ToTensorV2(),
    ]
)


# DataLoader를 위한 collate_fn. 이미지와 타겟을 리스트로 묶어줌
def collate_fn(batch):
    return tuple(zip(*batch))

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
        image_id = self.image_ids[idx]
        image_path = self.image_dir / image_id

        image = cv2.imread(str(image_path))
        if image is None:
            raise FileNotFoundError(
                f"Error: Could not load image at path: {image_path}"
            )

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w, _ = image.shape

        if self.mode in ["train", "val"]:
            records = self.df[self.df["file_name"] == image_id]
            boxes = records[["bbox_x", "bbox_y", "bbox_w", "bbox_h"]].values

            boxes[:, 2] = boxes[:, 0] + boxes[:, 2]
            boxes[:, 3] = boxes[:, 1] + boxes[:, 3]

            labels = records["label_idx"].values

            # print(f"\n[DEBUG 1] Image: {image_id}, Original Pixel Coords:\n{boxes}")

            # 바운딩 박스 좌표 정규화
            boxes = boxes.astype(np.float32)
            boxes[:, [0, 2]] /= w
            boxes[:, [1, 3]] /= h

            # print(f"[DEBUG 2] Normalized Coords for Albumentations:\n{boxes}")

            if self.transforms:
                try:
                    transformed = self.transforms(
                        image=image, bboxes=boxes, labels=labels
                    )
                    image = transformed["image"]
                    boxes = transformed["bboxes"]
                    labels = transformed["labels"]
                except Exception as e:
                    print(f"!!!!!!!!!!!!!! Albumentations에서 에러 발생 !!!!!!!!!!!!!!")
                    print(f"Image: {image_id}")
                    print(f"Boxes sent to transform: {boxes}")
                    # raise e  # 에러를 다시 발생시켜서 멈추게 함

            # ... 이하 코드는 이전과 동일 ...
            _, new_h, new_w = image.shape
            boxes = torch.as_tensor(boxes, dtype=torch.float32)

            if len(boxes) > 0:
                boxes[:, [0, 2]] *= new_w
                boxes[:, [1, 3]] *= new_h

            target = {
                "boxes": boxes,
                "labels": torch.as_tensor(labels, dtype=torch.int64),
            }

            if len(target["boxes"]) == 0:
                target["boxes"] = torch.zeros((0, 4), dtype=torch.float32)
                target["labels"] = torch.zeros((0,), dtype=torch.int64)

            return image, target

            # 테스트 모드일 경우, 이미지와 파일 이름만 반환
        elif self.mode == "test":
            # 테스트 시에는 보통 기본적인 리사이즈, 정규화만 적용
            if self.transforms:
                transformed = self.transforms(image=image)
                image = transformed["image"]

            # 나중에 예측 결과를 이미지와 매칭시키기 위해 파일 이름을 반환
            return image, image_id


# 참고: Subset을 사용할 때 transform을 다르게 적용하려면 약간의 트릭이 필요.
# 먼저 transform이 없는 전체 데이터셋을 만듦.
# 각 Subset에 맞는 transform을 적용하는 Wrapper 클래스 생성
# class TransformSubset(Dataset):
#     def __init__(self, subset, transforms):
#         self.subset = subset
#         self.transforms = transforms

#     def __getitem__(self, idx):
#         image, target = self.subset[idx]

#         # NumPy 배열로 변환 (Albumentations 입력 형식)
#         boxes = target["boxes"].numpy()
#         labels = target["labels"].numpy()

#         if self.transforms:
#             transformed = self.transforms(image=image, bboxes=boxes, labels=labels)
#             image = transformed["image"]
#             target["boxes"] = torch.as_tensor(
#                 transformed["bboxes"], dtype=torch.float32
#             )
#             # 증강 후 bbox가 사라졌을 경우 처리
#             if len(target["boxes"]) == 0:
#                 target["boxes"] = torch.zeros((0, 4), dtype=torch.float32)

#         return image, target

#     def __len__(self):
#         return len(self.subset)

import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

# pre-trained 모델 로드
# Faster R-CNN
# model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")

# MobileNetV3
model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_fpn(
    weights="DEFAULT"
)


# 분류기의 입력 피처 수를 가져옴
in_features = model.roi_heads.box_predictor.cls_score.in_features

# pre-trained head를 새로운 head로 교체
# num_classes에 배경(background) 클래스 1개를 더해줘야 함
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, NUM_CLASSES + 1)

import random


# 재현성을 위해 모든 난수 생성기의 시드를 고정하는 함수.
def seed_everything(seed: int):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"Seed set to {seed}")

    from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import Subset

seed_everything(SEED)

# 1. 데이터 준비
df = pd.read_csv(SAVE_PATH)

# StratifiedGroupKFold를 위한 데이터 준비
groups = df["file_name"]  # 그룹 기준: 이미지 파일 이름
labels = df["category_id"]  # 층화 기준: 원본 클래스 ID

# K-Fold 설정 (5-fold, 즉 80% train / 20% val)
cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)

# 첫 번째 fold의 train/val 인덱스를 가져옴
train_idxs, val_idxs = next(cv.split(df, labels, groups))
# 1. 인덱스를 사용해서 데이터프레임을 먼저 분할!
train_df_split = df.iloc[train_idxs].reset_index(drop=True)
val_df_split = df.iloc[val_idxs].reset_index(drop=True)

# 2. 분할된 데이터프레임으로 각각 Dataset 생성 (Subset, TransformSubset 불필요!)
train_dataset = PillDataset(
    df=train_df_split,
    image_dir=TRAIN_IMAGE_DIR,
    mode="train",
    transforms=train_transforms,
)

val_dataset = PillDataset(
    df=val_df_split,
    image_dir=TRAIN_IMAGE_DIR,
    mode="val",
    transforms=val_transforms,  # val_transforms 사용
)


test_df = pd.DataFrame({"file_name": os.listdir(TEST_IMAGE_DIR)})

test_dataset = PillDataset(
    df=test_df,
    image_dir=TEST_IMAGE_DIR,
    mode="test",
    transforms=test_transforms,
)


# --- Data Loader ---
train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    collate_fn=collate_fn,
)
val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    collate_fn=collate_fn,
)
test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    collate_fn=collate_fn,
)

img, image_id = next(iter(val_loader))
image_id[3]

model = model.to(DEVICE)

params = [p for p in model.parameters() if p.requires_grad]

# optimizer = torch.optim.SGD(
#     params,
#     lr=LEARNING_RATE,
#     momentum=MOMENTUM,
#     weight_decay=WEIGHT_DECAY,
# )
optimizer = torch.optim.AdamW(params, lr=LEARNING_RATE, weight_decay=0.01)

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=NUM_EPOCHS, eta_min=1e-6
)

for name, param in model.named_parameters():
    print(f"{name}: requires_grad={param.requires_grad}")

# 3. 학습 루프
print("--- Start Training ---")
metric = MeanAveragePrecision(box_format="xyxy").to(DEVICE)
# early_stopping = EarlyStopping(patience=7, verbose=True, path=path_model)
train_losses = []
val_losses = []
for epoch in range(NUM_EPOCHS):
    # ===================================
    #  Training Step
    # ===================================

    model.train()
    running_loss = 0.0
    loop = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{NUM_EPOCHS}]")

    for images, targets in loop:
        images = list(image.to(DEVICE) for image in images)
        targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()
        running_loss += losses.item()
    avg_train_loss = running_loss / len(train_loader)

    scheduler.step()

    # =====================================
    #  Validation Step (✨ 여기가 핵심 수정)
    # =====================================
    model.eval()
    val_loss = 0.0
    metric.reset()
    # (2) Validation phase
    for images, targets in val_loader:
        images = [img.to(DEVICE) for img in images]
        targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]
        # 1. mAP 계산을 위한 예측 (그래디언트 계산 불필요)
        with torch.no_grad():  # 👈✨ 예측 부분만 no_grad로 감싸기
            predictions = model(images)

        # 2. Metric 업데이트
        metric.update(predictions, targets)

        # 3. Validation Loss 계산 (그래디언트 계산 필요)
        #    torch.no_grad() 블록 바깥에서 계산
        model.train()  # Loss 계산을 위해 잠시 train 모드로
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        val_loss += losses.item()
        model.eval()  # 다음 배치를 위해 다시 eval 모드로 복귀

    average_val_loss = val_loss / len(val_loader)
    val_losses.append(average_val_loss)

    # mAP 평가
    mAP_dict = metric.compute()
    # mAP = evaluate_model(all_predictions, all_ground_truths, classes)
    print(
        f"Train Loss: {avg_train_loss:.4f}, Val Loss: {average_val_loss:.4f}, Validation mAP: {mAP_dict['map_50']:.4f}"
    )

print("--- Finish Training ---"),
# 최종 모델 저장
os.makedirs(EXPERIMENT_DIR, exist_ok=True)
torch.save(model.state_dict(), f"{EXPERIMENT_DIR}/final_model.pt")

print(EXPERIMENT_DIR)