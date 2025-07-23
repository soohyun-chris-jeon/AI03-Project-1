# ====================================================================
# model.py
# ====================================================================
# 모델의 아키텍처를 정의하는 곳
# pretrained 모델의 구조를 여기서 코드로 구현하거나 불러오게 됨.
# src/models/model.py

import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


def get_model(
    num_classes: int,
    model_name: str = "fasterrcnn_mobilenet_v3_large_fpn",
    pretrained: bool = True,
):
    """
    지정된 이름과 클래스 개수에 맞는 pre-trained object detection 모델을 반환.

    Args:
        num_classes (int): 타겟 클래스의 개수 (배경 포함).
        model_name (str): 불러올 모델의 이름.
        pretrained (bool): pre-trained 가중치를 사용할지 여부.

    Returns:
        torch.nn.Module: PyTorch 모델.
    """
    weights = "DEFAULT" if pretrained else None

    if model_name == "fasterrcnn_mobilenet_v3_large_fpn":
        model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_fpn(
            weights=weights
        )
    elif model_name == "fasterrcnn_resnet50_fpn":
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=weights)
    else:
        raise ValueError(f"지원하지 않는 모델 이름입니다: {model_name}")

    # 분류기(classifier)의 입력 피처 수를 가져옴
    in_features = model.roi_heads.box_predictor.cls_score.in_features

    # pre-trained head를 새로운 head로 교체
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model
