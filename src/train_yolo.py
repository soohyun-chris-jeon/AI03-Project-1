import config
from ultralytics import YOLO

model = YOLO("yolov8n.pt")

# YOLO 내장 학습 API에 증강 파라미터 전달
model.train(
    # --- 기본 학습 설정 ---
    data="YOLO_data.yaml",
    epochs=100,
    optimizer="AdamW",
    lr0=config.LEARNING_RATE,
    # --- 데이터 증강 설정 ---
    degrees=15.0,  # 15도 내에서 회전
    translate=0.1,  # 10% 내에서 이동
    scale=0.5,  # 50% ~ 150% 크기 조절
    fliplr=0.5,  # 50% 확률로 좌우 반전
    mosaic=1.0,  # Mosaic 증강 항상 사용
    mixup=0.1,  # Mixup 증강 10% 확률로 사용
    hsv_s=0.7,  # 채도 70% 내에서 변경
)
