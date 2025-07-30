# ====================================================================
# model.py
# ====================================================================
# 모델의 아키텍처를 정의하는 곳
# pretrained 모델의 구조를 여기서 코드로 구현하거나 불러오게 됨.
# src/models/model.py

import torchvision
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.rpn import AnchorGenerator


def get_model(
    num_classes: int,
    model_name: str = "mobilenet",
    pretrained: bool = True,
    anchor_sizes=None,
    aspect_ratios=None,
    min_size: int = 350,
    rpn_pre_nms_top_n_train: int = 2000,
    rpn_post_nms_top_n_train: int = 2000,
    rpn_pre_nms_top_n_test: int = 1000,
    rpn_post_nms_top_n_test: int = 1000,
):
    """
    지정된 이름과 클래스 개수에 맞는 pre-trained object detection 모델을 반환.

    Args:
        num_classes (int): 타겟 클래스 수 (배경 포함).
        model_name (str): 모델 이름 (e.g., fasterrcnn_resnet50_fpn).
        pretrained (bool): COCO pretrained weight 사용할지 여부.
        anchor_sizes (list): anchor 사이즈 세트.
        aspect_ratios (list): anchor 비율 세트.
        min_size (int): 이미지 입력 크기.
        rpn_pre_nms_top_n (int): RPN NMS 전 proposal 개수.
        rpn_post_nms_top_n (int): RPN NMS 후 proposal 개수.

    Returns:
        torch.nn.Module: PyTorch 모델.
    """
    weights = "DEFAULT" if pretrained else None
    # ====================================================================
    # 1. Baseline 코드 모델
    if model_name == "mobilenet":
        model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_fpn(
            weights=weights
        )

    # ====================================================================
    # 2. Faster-RCNN (resnet 기반)
    elif model_name == "resnet":
        if anchor_sizes is None:
            anchor_sizes = ((8,), (16,), (24,), (32,), (40,))
        if aspect_ratios is None:
            aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)
        anchor_generator = AnchorGenerator(anchor_sizes, aspect_ratios)

        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
            weights=FasterRCNN_ResNet50_FPN_Weights.COCO_V1,
            rpn_anchor_generator=anchor_generator,
            min_size=min_size,  # 작은 객체를 위한 이미지 크기
            box_batch_size_per_image=100,  # proposal 수 조절
        )

        # RPN proposal 개수 조정
        # model.rpn.pre_nms_top_n["training"] = rpn_pre_nms_top_n
        # model.rpn.post_nms_top_n["training"] = rpn_post_nms_top_n
        model.rpn.pre_nms_top_n_train = rpn_pre_nms_top_n_train
        model.rpn.post_nms_top_n_train = rpn_post_nms_top_n_train
        model.rpn.pre_nms_top_n_test = rpn_pre_nms_top_n_test
        model.rpn.post_nms_top_n_test = rpn_post_nms_top_n_test

    else:
        raise ValueError(f"지원하지 않는 모델 이름입니다: {model_name}")

    # 분류기(classifier)의 입력 피처 수를 가져옴
    in_features = model.roi_heads.box_predictor.cls_score.in_features

    # pre-trained head를 새로운 head로 교체
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model
