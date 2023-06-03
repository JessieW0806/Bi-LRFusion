import torch.nn as nn
import torch
import torch.nn.init as init
from torch.nn import functional as F
def init_conv(conv, glu=True):
    init.xavier_uniform_(conv.weight)
    if conv.bias is not None:
        conv.bias.data.zero_()

class SelfAttention(nn.Module):
    """
        Self attention Layer.
        Source paper: https://arxiv.org/abs/1805.08318
    """
    def __init__(self, in_dim, activation=F.relu):
        super(SelfAttention, self).__init__()
        self.chanel_in = in_dim
        self.activation = activation
        
        self.f = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8 , kernel_size=1).cuda()
        self.g = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8 , kernel_size=1).cuda()
        self.h = nn.Conv2d(in_channels=in_dim, out_channels=in_dim , kernel_size=1).cuda()
        
        self.gamma = nn.Parameter(torch.zeros(1)).cuda()

        self.softmax  = nn.Softmax(dim=-1).cuda()

        init_conv(self.f)
        init_conv(self.g)
        init_conv(self.h)
        
    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X W X H)
            returns :
                out : self attention feature maps
                
        """
        m_batchsize, C, width, height = x.size()
        
        f = self.f(x).view(m_batchsize, -1, width * height) # B * (C//8) * (W * H)
        g = self.g(x).view(m_batchsize, -1, width * height) # B * (C//8) * (W * H)
        h = self.h(x).view(m_batchsize, -1, width * height) # B * C * (W * H)
        #torch.cuda.empty_cache()
        attention = torch.bmm(f.permute(0, 2, 1), g).cuda() # B * (W * H) * (W * H)##4*16384*16384
        attention = self.softmax(attention)
        #torch.cuda.empty_cache()
        self_attetion = torch.bmm(h, attention).cuda() # B * C * (W * H) 
        #torch.cuda.empty_cache()
        self_attetion = self_attetion.view(m_batchsize, C, width, height) # B * C * W * H
        
        out = self.gamma * self_attetion + x
        return out