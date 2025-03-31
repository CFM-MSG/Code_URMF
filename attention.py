#论文： 残差注意力

import torch.nn as nn
import torch
import torch.nn.functional as F
import math
from torch.nn import init
class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


# class ChannelGate(nn.Module):
#     def __init__(self, gate_channel=3, reduction_ratio=16, num_layers=1):
#         super(ChannelGate, self).__init__()
#         # self.gate_activation = gate_activation
#         self.gate_c = nn.Sequential()
#         self.gate_c.add_module( 'flatten', Flatten() )
#         gate_channels = [gate_channel]
#         gate_channels += [gate_channel // reduction_ratio] * num_layers
#         gate_channels += [gate_channel]
#         for i in range( len(gate_channels) - 2 ):
#             self.gate_c.add_module( 'gate_c_fc_%d'%i, nn.Linear(gate_channels[i], gate_channels[i+1]) )
#             self.gate_c.add_module( 'gate_c_bn_%d'%(i+1), nn.BatchNorm1d(gate_channels[i+1]) )
#             self.gate_c.add_module( 'gate_c_relu_%d'%(i+1), nn.ReLU() )
#         self.gate_c.add_module( 'gate_c_fc_final', nn.Linear(gate_channels[-2], gate_channels[-1]) )
#     def forward(self, in_tensor):
#         avg_pool = F.avg_pool1d( in_tensor, in_tensor.size(2), stride=in_tensor.size(2) )
#         return self.gate_c( avg_pool ).unsqueeze(2).expand_as(in_tensor)

class ChannelGate(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max']):
        super(ChannelGate, self).__init__()
        self.gate_channels = gate_channels
        # self.mlp = nn.Sequential(
        #     Flatten(),
        #     nn.Linear(gate_channels, gate_channels // reduction_ratio),
        #     nn.ReLU(),
        #     nn.Linear(gate_channels // reduction_ratio, gate_channels)
        # )
        self.pool_types = pool_types

        self.con=nn.Conv1d(3, 3, kernel_size=3,stride=1,padding=1,bias=False)

        # init.kaiming_uniform_(self.mlp[1].weight, mode='fan_in', nonlinearity='relu')
        # init.kaiming_uniform_(self.mlp[3].weight, mode='fan_in', nonlinearity='relu')
        
        # init.xavier_uniform_(self.mlp[1].weight, gain=init.calculate_gain('relu'))
        # init.xavier_uniform_(self.mlp[3].weight, gain=init.calculate_gain('relu'))
    def forward(self, x):
        channel_att_sum = None
        # for pool_type in self.pool_types:
        if self.pool_types == 'avg':
            # Calculate average pool along the feature_dim dimension
            avg_pool = x.mean(dim=2)
            avg_pool =avg_pool.unsqueeze(2)
            channel_att_raw=self.con(avg_pool)
            # channel_att_raw = self.mlp(avg_pool)
        elif self.pool_types == 'max':
            # Calculate max pool along the feature_dim dimension
            max_pool = x.max(dim=2)
            channel_att_raw = self.mlp(max_pool)
        elif self.pool_types == 'lp':
            # Calculate Lp pool along the feature_dim dimension
            lp_pool = x.norm(2, dim=2)
            channel_att_raw = self.mlp(lp_pool)
        elif self.pool_types == 'lse':
            # LSE pool only
            lse_pool = F.logsumexp_1d(x, dim=2)
            channel_att_raw = self.mlp(lse_pool)

        if channel_att_sum is None:
            channel_att_sum = channel_att_raw
        else:
            channel_att_sum = channel_att_sum + channel_att_raw

        # scale = torch.sigmoid(channel_att_sum).unsqueeze(2).expand_as(x)
        score=avg_pool + channel_att_raw
        # x=x+x*scale
        x=x*score+x
        # fuison_output=torch.cat((x[:,0,:],x[:,1,:],x[:,2,:]),dim=1)
        fuison_output=(x[:,0,:]+x[:,1,:]+x[:,2,:])
        # fuison_output=x
        return fuison_output   
if __name__ =="__main__":
    a=torch.randn(50,3,72)
    fusion=ChannelGate(3,3,'avg')
    b=fusion(a)
    print(b)