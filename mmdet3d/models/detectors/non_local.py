import torch
import torch.nn as nn
from mmcv.cnn import constant_init, normal_init, ConvModule, build_conv_layer, kaiming_init
class NonLocal2D(nn.Module):
    """Non-local module.
    See https://arxiv.org/abs/1711.07971 for details.
    Args:
        in_channels (int): Channels of the input feature map.
        reduction (int): Channel reduction ratio.
        use_scale (bool): Whether to scale pairwise_weight by 1/inter_channels.
        conv_cfg (dict): The config dict for convolution layers.
            (only applicable to conv_out)
        norm_cfg (dict): The config dict for normalization layers.
            (only applicable to conv_out)
        mode (str): Options are `embedded_gaussian` and `dot_product`.
    """

    def __init__(self,
                 in_channels,
                 reduction=2,
                 use_scale=True,
                 conv_cfg=dict(type='Conv2d'),
                 norm_cfg=None,
                 sub_sample = True,
                 mode='dot_product'):
        super(NonLocal2D, self).__init__()
        self.in_channels = in_channels
        self.reduction = reduction
        self.use_scale = use_scale
        self.inter_channels = in_channels // reduction
        self.mode = mode
        assert mode in ['embedded_gaussian', 'dot_product']
        self.sub_sample = sub_sample
        # g, theta, phi are actually `nn.Conv2d`. Here we use ConvModule for
        # potential usage.
        
        self.g = ConvModule(
            self.in_channels,
            self.inter_channels,
            kernel_size=1,
            conv_cfg=dict(type='Conv2d'),
            act_cfg= None)
        
        self.theta = ConvModule(
            self.in_channels,
            self.inter_channels,
            kernel_size=1,
            conv_cfg=dict(type='Conv2d'),
            act_cfg = None)
        self.phi = ConvModule(
            self.in_channels,
            self.inter_channels,
            kernel_size=1,
            conv_cfg=dict(type='Conv2d'),
            act_cfg = None)

        if sub_sample:
            self.max_pool_layer = nn.MaxPool2d(kernel_size=(2, 2))
            #self.g = nn.Sequential(self.g, max_pool_layer)
            #self.phi = nn.Sequential(self.phi, max_pool_layer)
        self.conv_out = ConvModule(
            self.inter_channels,
            self.in_channels,
            kernel_size=1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg = None)
        #print(self.g)
        #print(self.theta)
        ##print(self.conv_out)
        #exit()
        self.conv_mask = nn.Conv2d(in_channels=self.inter_channels, out_channels=in_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.gamma = nn.Parameter(torch.zeros(1)).cuda()
        self.init_weights()

    def init_weights(self, std=0.01, zeros_init=False):
        for m in [self.g, self.theta, self.phi]:
            normal_init(m.conv, std=std)
        if zeros_init:
            constant_init(self.conv_out.conv, 0)
        else:
            normal_init(self.conv_out.conv, std=std) #self.conv_out.norm_cfg is None:

    def embedded_gaussian(self, theta_x, phi_x):
        # pairwise_weight: [N, HxW, HxW]
        pairwise_weight = torch.matmul(theta_x, phi_x)
        if self.use_scale:
            # theta_x.shape[-1] is `self.inter_channels`
            pairwise_weight /= theta_x.shape[-1]**-0.5
        pairwise_weight = pairwise_weight.softmax(dim=-1)
        return pairwise_weight

    def dot_product(self, theta_x, phi_x):
        # pairwise_weight: [N, HxW, HxW]
        pairwise_weight = torch.matmul(theta_x, phi_x)
        pairwise_weight /= pairwise_weight.shape[-1]
        return pairwise_weight

    def forward(self, x, x2):
        n, _, h, w = x.shape
        #print(x.shape)
        # g_x: [N, HxW, C]
        
        g_x2 = self.max_pool_layer(self.g(x2)).view(n, self.inter_channels, -1)
        g_x2 = g_x2.permute(0, 2, 1)
        #print(g_x2.shape)
        # theta_x: [N, HxW, C]
        theta_x = self.theta(x).view(n, self.inter_channels, -1)
        theta_x = theta_x.permute(0, 2, 1)
        #print(theta_x.shape)
        # phi_x: [N, C, HxW]
        phi_x2 = self.max_pool_layer(self.phi(x2)).view(n, self.inter_channels, -1)
        #print(phi_x2.shape)
        pairwise_func = getattr(self, self.mode)
        # pairwise_weight: [N, HxW, HxW]
        pairwise_weight = pairwise_func(theta_x, phi_x2)
        #print(phi_x2)
        #exit()
        # y: [N, HxW, C]
        y = torch.matmul(pairwise_weight, g_x2)
        # y: [N, C, H, W]
        y = y.permute(0, 2, 1).reshape(n, self.inter_channels, h, w)
        mask = self.conv_out(y)
        output = x + mask 
        #output = x + self.gamma * y
        return output