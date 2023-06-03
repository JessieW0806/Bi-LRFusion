import torch.nn as nn
import torch
import torch.nn.init as init
from torch.nn import functional as F
class ChannelAttentionModule(nn.Module):
    def __init__(self, channel, ratio=16):
        super(ChannelAttentionModule, self).__init__()
        #使用自适应池化缩减map的大小，保持通道不变
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
 
        self.shared_MLP = nn.Sequential(
            nn.Conv2d(channel, channel // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(channel // ratio, channel, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
 
    def forward(self, x):
        avgout = self.shared_MLP(self.avg_pool(x))
        maxout = self.shared_MLP(self.max_pool(x))
        return self.sigmoid(avgout + maxout)



class SpatialGate(nn.Module):
    def __init__(self):
        super(SpatialGate, self).__init__()
        kernel_size = 7
        #self.compress = ChannelPool()
        self.spatial = nn.Conv2d(2, 1, kernel_size, stride=1, padding=(kernel_size-1) // 2, bias=False)
    def forward(self, bev, rgb):
        #bev_compress = self.compress(bev)
        avg_out = torch.mean(bev, dim=1, keepdim=True)
        max_out, _ = torch.max(bev, dim=1, keepdim=True)
        bev_compress = torch.cat([avg_out, max_out], dim=1)
        bev_out = self.spatial(bev_compress)
        scale = torch.sigmoid(bev_out) # broadcasting
        return rgb*scale

