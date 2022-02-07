import copy
from .resnet import ResNet
from .ghostnet import GhostNet
from .shufflenetv2 import ShuffleNetV2
from .shufflenetv2_preprocessing import ShuffleNetV2_preprocessing
from .mobilenetv2 import MobileNetV2
from .efficientnet_lite import EfficientNetLite
from .custom_csp import CustomCspNet
from .repvgg import RepVGG
from .convnext import ConvNeXT


def build_backbone(cfg):
    backbone_cfg = copy.deepcopy(cfg)
    name = backbone_cfg.pop('name')
    if name == 'ResNet':
        return ResNet(**backbone_cfg)
    elif name == 'ShuffleNetV2':
        return ShuffleNetV2(**backbone_cfg)
    elif name == 'ShuffleNetV2_preprocessing':
        return ShuffleNetV2_preprocessing(**backbone_cfg)
    elif name == 'GhostNet':
        return GhostNet(**backbone_cfg)
    elif name == 'MobileNetV2':
        return MobileNetV2(**backbone_cfg)
    elif name == 'EfficientNetLite':
        return EfficientNetLite(**backbone_cfg)
    elif name == 'CustomCspNet':
        return CustomCspNet(**backbone_cfg)
    elif name == 'RepVGG':
        return RepVGG(**backbone_cfg)
    elif name == 'ConvNext':
        return ConvNeXT(**backbone_cfg)
    else:
        raise NotImplementedError

