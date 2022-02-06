"""
ConvModule refers from MMDetection
RepVGGConvModule refers from RepVGG: Making VGG-style ConvNets Great Again
"""
import torch
import torch.nn as nn
import numpy as np
import warnings
import torch.nn.intrinsic as nni
from .init_weights import kaiming_init, normal_init, xavier_init, constant_init
from .norm import build_norm_layer
from .activation import act_layers
import cv2
import torchvision.transforms.functional as T

class ConvModule(nn.Module):
    """A conv block that contains conv/norm/activation layers.

    Args:
        in_channels (int): Same as nn.Conv2d.
        out_channels (int): Same as nn.Conv2d.
        kernel_size (int or tuple[int]): Same as nn.Conv2d.
        stride (int or tuple[int]): Same as nn.Conv2d.
        padding (int or tuple[int]): Same as nn.Conv2d.
        dilation (int or tuple[int]): Same as nn.Conv2d.
        groups (int): Same as nn.Conv2d.
        bias (bool or str): If specified as `auto`, it will be decided by the
            norm_cfg. Bias will be set as True if norm_cfg is None, otherwise
            False.
        conv_cfg (dict): Config dict for convolution layer.
        norm_cfg (dict): Config dict for normalization layer.
        activation (str): activation layer, "ReLU" by default.
        inplace (bool): Whether to use inplace mode for activation.
        order (tuple[str]): The order of conv/norm/activation layers. It is a
            sequence of "conv", "norm" and "act". Examples are
            ("conv", "norm", "act") and ("act", "conv", "norm").
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 bias='auto',
                 conv_cfg=None,
                 norm_cfg=None,
                 activation='ReLU',
                 inplace=True,
                 order=('conv', 'norm', 'act')):
        super(ConvModule, self).__init__()
        assert conv_cfg is None or isinstance(conv_cfg, dict)
        assert norm_cfg is None or isinstance(norm_cfg, dict)
        assert activation is None or isinstance(activation, str)
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.activation = activation
        self.inplace = inplace
        self.order = order
        assert isinstance(self.order, tuple) and len(self.order) == 3
        assert set(order) == set(['conv', 'norm', 'act'])

        self.with_norm = norm_cfg is not None
        # if the conv layer is before a norm layer, bias is unnecessary.
        if bias == 'auto':
            bias = False if self.with_norm else True
        self.with_bias = bias

        if self.with_norm and self.with_bias:
            warnings.warn('ConvModule has norm and bias at the same time')

        # build convolution layer
        self.conv = nn.Conv2d(  #
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias)
        # export the attributes of self.conv to a higher level for convenience
        self.in_channels = self.conv.in_channels
        self.out_channels = self.conv.out_channels
        self.kernel_size = self.conv.kernel_size
        self.stride = self.conv.stride
        self.padding = self.conv.padding
        self.dilation = self.conv.dilation
        self.transposed = self.conv.transposed
        self.output_padding = self.conv.output_padding
        self.groups = self.conv.groups

        # build normalization layers
        if self.with_norm:
            # norm layer is after conv layer
            if order.index('norm') > order.index('conv'):
                norm_channels = out_channels
            else:
                norm_channels = in_channels
            self.norm_name, norm = build_norm_layer(norm_cfg, norm_channels)
            self.add_module(self.norm_name, norm)
        else:
            self.norm_name = None

        # build activation layer
        if self.activation:
            self.act = act_layers(self.activation)

        # Use msra init by default
        self.init_weights()

    @property
    def norm(self):
        if self.norm_name:
            return getattr(self, self.norm_name)
        else:
            return None

    def init_weights(self):
        if self.activation == 'LeakyReLU':
            nonlinearity = 'leaky_relu'
        else:
            nonlinearity = 'relu'
        kaiming_init(self.conv, nonlinearity=nonlinearity)
        if self.with_norm:
            constant_init(self.norm, 1, bias=0)

    def forward(self, x, norm=True):
        for layer in self.order:
            if layer == 'conv':
                x = self.conv(x)
            elif layer == 'norm' and norm and self.with_norm:
                x = self.norm(x)
            elif layer == 'act' and self.activation:
                x = self.act(x)
        return x


class DepthwiseConvModule(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 bias='auto',
                 norm_cfg=dict(type='BN'),
                 activation='ReLU',
                 inplace=True,
                 order=('depthwise', 'dwnorm', 'act', 'pointwise', 'pwnorm', 'act')):
        super(DepthwiseConvModule, self).__init__()
        assert activation is None or isinstance(activation, str)
        self.activation = activation
        self.inplace = inplace
        self.order = order
        assert isinstance(self.order, tuple) and len(self.order) == 6
        assert set(order) == set(['depthwise', 'dwnorm', 'act', 'pointwise', 'pwnorm', 'act'])

        self.with_norm = norm_cfg is not None
        # if the conv layer is before a norm layer, bias is unnecessary.
        if bias == 'auto':
            bias = False if self.with_norm else True
        self.with_bias = bias

        if self.with_norm and self.with_bias:
            warnings.warn('ConvModule has norm and bias at the same time')

        # build convolution layer
        self.depthwise = nn.Conv2d(in_channels,
                                   in_channels,
                                   kernel_size,
                                   stride=stride,
                                   padding=padding,
                                   dilation=dilation,
                                   groups=in_channels,
                                   bias=bias)
        self.pointwise = nn.Conv2d(in_channels,
                                   out_channels,
                                   kernel_size=1,
                                   stride=1,
                                   padding=0,
                                   bias=bias)

        # export the attributes of self.conv to a higher level for convenience
        self.in_channels = self.depthwise.in_channels
        self.out_channels = self.pointwise.out_channels
        self.kernel_size = self.depthwise.kernel_size
        self.stride = self.depthwise.stride
        self.padding = self.depthwise.padding
        self.dilation = self.depthwise.dilation
        self.transposed = self.depthwise.transposed
        self.output_padding = self.depthwise.output_padding

        # build normalization layers
        if self.with_norm:
            # norm layer is after conv layer
            _, self.dwnorm = build_norm_layer(norm_cfg, in_channels)
            _, self.pwnorm = build_norm_layer(norm_cfg, out_channels)

        # build activation layer
        if self.activation:
            self.act = act_layers(self.activation)

        # Use msra init by default
        self.init_weights()

    def init_weights(self):
        if self.activation == 'LeakyReLU':
            nonlinearity = 'leaky_relu'
        else:
            nonlinearity = 'relu'
        kaiming_init(self.depthwise, nonlinearity=nonlinearity)
        kaiming_init(self.pointwise, nonlinearity=nonlinearity)
        if self.with_norm:
            constant_init(self.dwnorm, 1, bias=0)
            constant_init(self.pwnorm, 1, bias=0)

    def forward(self, x, norm=True):
        for layer_name in self.order:
            if layer_name != 'act':
                layer = self.__getattr__(layer_name)
                x = layer(x)
            elif layer_name == 'act' and self.activation:
                x = self.act(x)
        return x
    def qat(self):
        self.depthwise= nni.ConvBnReLU2d(self.depthwise,self.dwnorm,self.act)
        self.pointwise= nni.ConvBnReLU2d(self.pointwise,self.dwnorm,self.act)
        self.dwnorm= torch.nn.Identity()
        self.pwnorm= torch.nn.Identity()
        self.act= torch.nn.Identity()


class RepVGGConvModule(nn.Module):
    """
    RepVGG Conv Block from paper RepVGG: Making VGG-style ConvNets Great Again
    https://arxiv.org/abs/2101.03697
    https://github.com/DingXiaoH/RepVGG
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 activation='ReLU',
                 padding_mode='zeros',
                 deploy=False):
        super(RepVGGConvModule, self).__init__()
        assert activation is None or isinstance(activation, str)
        self.activation = activation

        self.deploy = deploy
        self.groups = groups
        self.in_channels = in_channels

        assert kernel_size == 3
        assert padding == 1

        padding_11 = padding - kernel_size // 2

        # build activation layer
        if self.activation:
            self.act = act_layers(self.activation)

        if deploy:
            self.rbr_reparam = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                         kernel_size=kernel_size, stride=stride,
                                         padding=padding, dilation=dilation, groups=groups, bias=True,
                                         padding_mode=padding_mode)

        else:
            self.rbr_identity = nn.BatchNorm2d(
                num_features=in_channels) if out_channels == in_channels and stride == 1 else None

            self.rbr_dense = nn.Sequential(nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                                     kernel_size=kernel_size, stride=stride, padding=padding,
                                                     groups=groups, bias=False),
                                           nn.BatchNorm2d(num_features=out_channels))

            self.rbr_1x1 = nn.Sequential(nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                                   kernel_size=1, stride=stride, padding=padding_11,
                                                   groups=groups, bias=False),
                                         nn.BatchNorm2d(num_features=out_channels))
            print('RepVGG Block, identity = ', self.rbr_identity)

    def forward(self, inputs):
        if hasattr(self, 'rbr_reparam'):
            return self.act(self.rbr_reparam(inputs))

        if self.rbr_identity is None:
            id_out = 0
        else:
            id_out = self.rbr_identity(inputs)

        return self.act(self.rbr_dense(inputs) + self.rbr_1x1(inputs) + id_out)

    #   This func derives the equivalent kernel and bias in a DIFFERENTIABLE way.
    #   You can get the equivalent kernel and bias at any time and do whatever you want,
    #   for example, apply some penalties or constraints during training, just like you do to the other models.
    #   May be useful for quantization or pruning.
    def get_equivalent_kernel_bias(self):
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.rbr_dense)
        kernel1x1, bias1x1 = self._fuse_bn_tensor(self.rbr_1x1)
        kernelid, biasid = self._fuse_bn_tensor(self.rbr_identity)
        return kernel3x3 + self._pad_1x1_to_3x3_tensor(kernel1x1) + kernelid, bias3x3 + bias1x1 + biasid

    def _pad_1x1_to_3x3_tensor(self, kernel1x1):
        if kernel1x1 is None:
            return 0
        else:
            return nn.functional.pad(kernel1x1, [1, 1, 1, 1])

    def _fuse_bn_tensor(self, branch):
        if branch is None:
            return 0, 0
        if isinstance(branch, nn.Sequential):
            kernel = branch[0].weight
            running_mean = branch[1].running_mean
            running_var = branch[1].running_var
            gamma = branch[1].weight
            beta = branch[1].bias
            eps = branch[1].eps
        else:
            assert isinstance(branch, nn.BatchNorm2d)
            if not hasattr(self, 'id_tensor'):
                input_dim = self.in_channels // self.groups
                kernel_value = np.zeros((self.in_channels, input_dim, 3, 3), dtype=np.float32)
                for i in range(self.in_channels):
                    kernel_value[i, i % input_dim, 1, 1] = 1
                self.id_tensor = torch.from_numpy(kernel_value).to(branch.weight.device)
            kernel = self.id_tensor
            running_mean = branch.running_mean
            running_var = branch.running_var
            gamma = branch.weight
            beta = branch.bias
            eps = branch.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std

    def repvgg_convert(self):
        kernel, bias = self.get_equivalent_kernel_bias()
        return kernel.detach().cpu().numpy(), bias.detach().cpu().numpy(),


class AutoAug(nn.Module):
    def __init__(self):
        super(AutoAug, self).__init__()
        self.mean = torch.from_numpy( np.array([127.0], dtype=np.float32).reshape(1, 1, 1) / 255).cuda()
        self.std =  torch.from_numpy(np.array([128.0], dtype=np.float32).reshape(1, 1, 1) / 255).cuda()
        
        self.a1 = torch.nn.Parameter(torch.rand(()))
        self.a2 = torch.nn.Parameter(torch.rand(()))
        self.a3 = torch.nn.Parameter(torch.rand(()))
        self.a4 = torch.nn.Parameter(torch.rand(()))
        self.a5 = torch.nn.Parameter(torch.rand(()))
        self.a6 = torch.nn.Parameter(torch.rand(()))
        self.a7 = torch.nn.Parameter(torch.rand(()))
        self.a8 = torch.nn.Parameter(torch.rand(()))
        self.a9 = torch.nn.Parameter(torch.rand(()))
        self.a10 = torch.nn.Parameter(torch.rand(()))
        self.a11 = torch.nn.Parameter(torch.rand(()))
#        self.a12 = torch.nn.Parameter(torch.rand(()))
        
        self.b0 = torch.nn.Parameter(torch.rand(()))
        self.b1 = torch.nn.Parameter(torch.rand(()))
        self.b2 = torch.nn.Parameter(torch.rand(()))
        self.b3 = torch.nn.Parameter(torch.rand(()))
        self.b4 = torch.nn.Parameter(torch.rand(()))
        self.b5 = torch.nn.Parameter(torch.rand(()))
        self.b6 = torch.nn.Parameter(torch.rand(()))
        self.b7 = torch.nn.Parameter(torch.rand(()))
        self.b8 = torch.nn.Parameter(torch.rand(()))
        self.b9 = torch.nn.Parameter(torch.rand(()))
        self.b10 = torch.nn.Parameter(torch.rand(()))
        self.b11 = torch.nn.Parameter(torch.rand(()))
#        self.b12 = torch.nn.Parameter(torch.rand(()))

        self.sum_b = torch.sigmoid(self.b0)+torch.sigmoid(self.b1)+torch.sigmoid(self.b2)+torch.sigmoid(self.b3)+torch.sigmoid(self.b4)+torch.sigmoid(self.b5)+torch.sigmoid(self.b6)+torch.sigmoid(self.b7)+torch.sigmoid(self.b8)+torch.sigmoid(self.b9)+torch.sigmoid(self.b10)+torch.sigmoid(self.b11)#
        
    def blur(self,x):
        out= T.gaussian_blur(x,kernel_size=(5, 5))
        return out
    
    def sharpening(self,x,factor=2):
        out= T.adjust_sharpness(x,sharpness_factor=factor)
        return out
    
    def histo_equ(self,x):
        out= T.equalize(x.to(dtype=torch.uint8))
        return out.to(dtype=torch.float32)
    
    def adj_brightness(self,x,factor=0.1):
        out= T.adjust_brightness(x,brightness_factor=factor)
        return out
  
    def adj_contrast(self,x,factor=2):
        out= T.adjust_contrast(x,contrast_factor=factor)
        return out

    def auto_contrast(self,x):
        out= T.autocontrast(x)
        return out

    def adj_gamma(self,x,factor=0.1):
        out= T.adjust_gamma(x,gamma=factor)
        return out
    
#    def log_cor(self,x):
#        c = 255 / torch.log(1 + torch.max(x))
#        out = c * (torch.log(x + 1))
#        return out

    
    def forward(self, x):
        x = x*self.std+self.mean
        out1= self.blur(x)
        out2= self.sharpening(x,factor=0)
        out3= self.sharpening(x,factor=2)
        out4= self.histo_equ(x)
        out5= self.adj_brightness(x,factor=0.1)
        out6= self.adj_brightness(x,factor=2)
        out7= self.adj_contrast(x,factor=0.1)
        out8= self.adj_contrast(x,factor=2)
        out9= self.auto_contrast(x)
        out10= self.adj_gamma(x,factor=0.1)
        out11= self.adj_gamma(x,factor=2)
#        out12= self.log_cor(x)
        
#        x = torch.sigmoid(self.b0)*x+torch.sigmoid(self.b1)*(torch.sigmoid(self.a1)*x+(1-torch.sigmoid(self.a1))*out1)+torch.sigmoid(self.b2)*(torch.sigmoid(self.a2)*x+(1-torch.sigmoid(self.a2))*out2)+torch.sigmoid(self.b3)*(torch.sigmoid(self.a3)*x+(1-torch.sigmoid(self.a3))*out3)+torch.sigmoid(self.b4)*(torch.sigmoid(self.a4)*x+(1-torch.sigmoid(self.a4))*out4)+torch.sigmoid(self.b5)*(torch.sigmoid(self.a5)*x+(1-torch.sigmoid(self.a5))*out5)+torch.sigmoid(self.b6)*(torch.sigmoid(self.a6)*x+(1-torch.sigmoid(self.a6))*out6)+torch.sigmoid(self.b7)*(torch.sigmoid(self.a7)*x+(1-torch.sigmoid(self.a7))*out7)+torch.sigmoid(self.b8)*(torch.sigmoid(self.a8)*x+(1-torch.sigmoid(self.a8))*out8)+torch.sigmoid(self.b9)*(torch.sigmoid(self.a9)*x+(1-torch.sigmoid(self.a9))*out9)+torch.sigmoid(self.b10)*(torch.sigmoid(self.a10)*x+(1-torch.sigmoid(self.a10))*out10)+torch.sigmoid(self.b11)*(torch.sigmoid(self.a11)*x+(1-torch.sigmoid(self.a11))*out11)
        
        x = torch.sigmoid(self.b0)/(torch.sigmoid(self.b0)+torch.sigmoid(self.b1)+torch.sigmoid(self.b2)+torch.sigmoid(self.b3)+torch.sigmoid(self.b4)+torch.sigmoid(self.b5)+torch.sigmoid(self.b6)+torch.sigmoid(self.b7)+torch.sigmoid(self.b8)+torch.sigmoid(self.b9)+torch.sigmoid(self.b10)+torch.sigmoid(self.b11))*x+torch.sigmoid(self.b1)/(torch.sigmoid(self.b0)+torch.sigmoid(self.b1)+torch.sigmoid(self.b2)+torch.sigmoid(self.b3)+torch.sigmoid(self.b4)+torch.sigmoid(self.b5)+torch.sigmoid(self.b6)+torch.sigmoid(self.b7)+torch.sigmoid(self.b8)+torch.sigmoid(self.b9)+torch.sigmoid(self.b10)+torch.sigmoid(self.b11))*(torch.sigmoid(self.a1)*x+(1-torch.sigmoid(self.a1))*out1)+torch.sigmoid(self.b2)/(torch.sigmoid(self.b0)+torch.sigmoid(self.b1)+torch.sigmoid(self.b2)+torch.sigmoid(self.b3)+torch.sigmoid(self.b4)+torch.sigmoid(self.b5)+torch.sigmoid(self.b6)+torch.sigmoid(self.b7)+torch.sigmoid(self.b8)+torch.sigmoid(self.b9)+torch.sigmoid(self.b10)+torch.sigmoid(self.b11))*(torch.sigmoid(self.a2)*x+(1-torch.sigmoid(self.a2))*out2)+torch.sigmoid(self.b3)/(torch.sigmoid(self.b0)+torch.sigmoid(self.b1)+torch.sigmoid(self.b2)+torch.sigmoid(self.b3)+torch.sigmoid(self.b4)+torch.sigmoid(self.b5)+torch.sigmoid(self.b6)+torch.sigmoid(self.b7)+torch.sigmoid(self.b8)+torch.sigmoid(self.b9)+torch.sigmoid(self.b10)+torch.sigmoid(self.b11))*(torch.sigmoid(self.a3)*x+(1-torch.sigmoid(self.a3))*out3)+torch.sigmoid(self.b4)/(torch.sigmoid(self.b0)+torch.sigmoid(self.b1)+torch.sigmoid(self.b2)+torch.sigmoid(self.b3)+torch.sigmoid(self.b4)+torch.sigmoid(self.b5)+torch.sigmoid(self.b6)+torch.sigmoid(self.b7)+torch.sigmoid(self.b8)+torch.sigmoid(self.b9)+torch.sigmoid(self.b10)+torch.sigmoid(self.b11))*(torch.sigmoid(self.a4)*x+(1-torch.sigmoid(self.a4))*out4)+torch.sigmoid(self.b5)/(torch.sigmoid(self.b0)+torch.sigmoid(self.b1)+torch.sigmoid(self.b2)+torch.sigmoid(self.b3)+torch.sigmoid(self.b4)+torch.sigmoid(self.b5)+torch.sigmoid(self.b6)+torch.sigmoid(self.b7)+torch.sigmoid(self.b8)+torch.sigmoid(self.b9)+torch.sigmoid(self.b10)+torch.sigmoid(self.b11))*(torch.sigmoid(self.a5)*x+(1-torch.sigmoid(self.a5))*out5)+torch.sigmoid(self.b6)/(torch.sigmoid(self.b0)+torch.sigmoid(self.b1)+torch.sigmoid(self.b2)+torch.sigmoid(self.b3)+torch.sigmoid(self.b4)+torch.sigmoid(self.b5)+torch.sigmoid(self.b6)+torch.sigmoid(self.b7)+torch.sigmoid(self.b8)+torch.sigmoid(self.b9)+torch.sigmoid(self.b10)+torch.sigmoid(self.b11))*(torch.sigmoid(self.a6)*x+(1-torch.sigmoid(self.a6))*out6)+torch.sigmoid(self.b7)/(torch.sigmoid(self.b0)+torch.sigmoid(self.b1)+torch.sigmoid(self.b2)+torch.sigmoid(self.b3)+torch.sigmoid(self.b4)+torch.sigmoid(self.b5)+torch.sigmoid(self.b6)+torch.sigmoid(self.b7)+torch.sigmoid(self.b8)+torch.sigmoid(self.b9)+torch.sigmoid(self.b10)+torch.sigmoid(self.b11))*(torch.sigmoid(self.a7)*x+(1-torch.sigmoid(self.a7))*out7)+torch.sigmoid(self.b8)/(torch.sigmoid(self.b0)+torch.sigmoid(self.b1)+torch.sigmoid(self.b2)+torch.sigmoid(self.b3)+torch.sigmoid(self.b4)+torch.sigmoid(self.b5)+torch.sigmoid(self.b6)+torch.sigmoid(self.b7)+torch.sigmoid(self.b8)+torch.sigmoid(self.b9)+torch.sigmoid(self.b10)+torch.sigmoid(self.b11))*(torch.sigmoid(self.a8)*x+(1-torch.sigmoid(self.a8))*out8)+torch.sigmoid(self.b9)/(torch.sigmoid(self.b0)+torch.sigmoid(self.b1)+torch.sigmoid(self.b2)+torch.sigmoid(self.b3)+torch.sigmoid(self.b4)+torch.sigmoid(self.b5)+torch.sigmoid(self.b6)+torch.sigmoid(self.b7)+torch.sigmoid(self.b8)+torch.sigmoid(self.b9)+torch.sigmoid(self.b10)+torch.sigmoid(self.b11))*(torch.sigmoid(self.a9)*x+(1-torch.sigmoid(self.a9))*out9)+torch.sigmoid(self.b10)/(torch.sigmoid(self.b0)+torch.sigmoid(self.b1)+torch.sigmoid(self.b2)+torch.sigmoid(self.b3)+torch.sigmoid(self.b4)+torch.sigmoid(self.b5)+torch.sigmoid(self.b6)+torch.sigmoid(self.b7)+torch.sigmoid(self.b8)+torch.sigmoid(self.b9)+torch.sigmoid(self.b10)+torch.sigmoid(self.b11))*(torch.sigmoid(self.a10)*x+(1-torch.sigmoid(self.a10))*out10)+torch.sigmoid(self.b11)/(torch.sigmoid(self.b0)+torch.sigmoid(self.b1)+torch.sigmoid(self.b2)+torch.sigmoid(self.b3)+torch.sigmoid(self.b4)+torch.sigmoid(self.b5)+torch.sigmoid(self.b6)+torch.sigmoid(self.b7)+torch.sigmoid(self.b8)+torch.sigmoid(self.b9)+torch.sigmoid(self.b10)+torch.sigmoid(self.b11))*(torch.sigmoid(self.a11)*x+(1-torch.sigmoid(self.a11))*out11)

        x= (x-self.mean)/self.std
        
        return x
        


