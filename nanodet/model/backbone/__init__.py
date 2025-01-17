import copy
from .resnet import ResNet
from .resnet1_stageratio import ResNet_1
from .resnet2_patchify import ResNet_2
from .resnet3_resnextify import ResNet_3
from .resnet4_invbottlenek import ResNet_4
from .resnet5_kernel import ResNet_5
from .resnet6_gelu import ResNet_6
from .resnet7_lessact import ResNet_7
from .resnet8_lessnorm import ResNet_8
from .resnet9_ln import ResNet_9

from .resnet_cust_1_resnextify import ResNet_cus_1
from .resnet_cust_2_lessnorm import ResNet_cus_2
from .resnet_cust_3_ln import ResNet_cus_3

from .ghostnet import GhostNet
from .shufflenetv2 import ShuffleNetV2
from .shufflenetv2_preprocessing import ShuffleNetV2_preprocessing
from .mobilenetv2 import MobileNetV2
from .efficientnet_lite import EfficientNetLite
from .custom_csp import CustomCspNet
from .repvgg import RepVGG
from .convnext_det import ConvNeXt_det
from .convnext import ConvNeXt


def build_backbone(cfg):
    backbone_cfg = copy.deepcopy(cfg)
    name = backbone_cfg.pop('name')
    if name == 'ResNet':
        return ResNet(**backbone_cfg)
        
    elif name == 'ResNet_1':
        return ResNet_1(**backbone_cfg)
    elif name == 'ResNet_2':
        return ResNet_2(**backbone_cfg)
    elif name == 'ResNet_3':
        return ResNet_3(**backbone_cfg)        
    elif name == 'ResNet_4':
        return ResNet_4(**backbone_cfg)   
    elif name == 'ResNet_5':
        return ResNet_5(**backbone_cfg)        
    elif name == 'ResNet_6':
        return ResNet_6(**backbone_cfg) 
    elif name == 'ResNet_7':
        return ResNet_7(**backbone_cfg)   
    elif name == 'ResNet_8':
        return ResNet_8(**backbone_cfg)        
    elif name == 'ResNet_9':
        return ResNet_9(**backbone_cfg)
        
    elif name == 'ResNet_cus_1':
        return ResNet_cus_1(**backbone_cfg)
    elif name == 'ResNet_cus_2':
        return ResNet_cus_2(**backbone_cfg) 
    elif name == 'ResNet_cus_3':
        return ResNet_cus_3(**backbone_cfg) 
        
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
        return ConvNeXt(**backbone_cfg)
    elif name == 'ConvNext_det':
        return ConvNeXt_det(**backbone_cfg)
    else:
        raise NotImplementedError

