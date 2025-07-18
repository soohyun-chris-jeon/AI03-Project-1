# ====================================================================
# Configuration 설정
# ====================================================================
# 하이퍼파라미터 및 경로 등 실험에 필요한 설정들을 모아둠
# 실험 추적 및 재현성을 위해 모든 값은 여기에서 수정하고자 함
import os  # 디렉토리, 파일 경로 조작 등

# 디바이스 설정
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# 주요 하이퍼파라미터
LEARNING_RATE = 1e-4  # 학습률 (optimizer용)
BATCH_SIZE = 16  # 배치 크기
NUM_EPOCHS = 100  # 학습 epoch 수
SEED = 42  # 재현성을 위한 random seed

# 데이터 경로 설정
# DATA_ROOT = path
DATA_ROOT = "../data/raw/ai03-level1-project/"  # 경로
train_image_dir = os.path.join(DATA_ROOT, "train_images")
train_ann_dir = os.path.join(DATA_ROOT, "train_annotations")
test_image_dir = os.path.join(DATA_ROOT, "test_images")

# 모델 설정
MODEL_NAME = "UNET"  # 또는 "EfficientNet", 등등
USE_PRETRAINED = True  # torchvision 모델 사용 여부

# 학습 고도화 설정 (Optional)
USE_SCHEDULER = True  # Learning rate scheduler 사용 여부
EARLY_STOPPING = True  # Early stopping 적용 여부
AUGMENTATION = True  # 데이터 증강 사용 여부

# 실험 로깅용 설정
USE_WANDB = True
WANDB_PROJECT = "AI03-Project-1"
RUN_NAME = f"{MODEL_NAME}_bs{BATCH_SIZE}_lr{LEARNING_RATE}"
