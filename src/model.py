# ====================================================================
# model.py
# ====================================================================
# 모델의 아키텍처를 정의하는 곳
# pretrained 모델의 구조를 여기서 코드로 구현하거나 불러오게 됨.

import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


def get_model(num_classes):
    """
    지정된 클래스 수에 맞게 수정된 Faster R-CNN 모델을 반환합니다.
    """
    # pre-trained 모델 로드
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")

    # 분류기의 입력 피처 수를 가져옴
    in_features = model.roi_heads.box_predictor.cls_score.in_features

    # pre-trained head를 새로운 head로 교체
    # num_classes에 배경(background) 클래스 1개를 더해줘야 함
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes + 1)

    return model
