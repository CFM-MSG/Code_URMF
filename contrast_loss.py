import torch
import torch.nn as nn
import torch.nn.functional as F

class Contrastive_loss(nn.Module):
    def __init__(self,tau):
        super(Contrastive_loss,self).__init__()
        self.tau=tau

    def sim(self,z1:torch.Tensor,z2:torch.Tensor):
        z1 = F.normalize(z1)
        z2 = F.normalize(z2)
        return torch.mm(z1,z2.t())
    
    def semi_loss(self,z1:torch.Tensor,z2:torch.Tensor):
        f=lambda x: torch.exp(x/self.tau)
        refl_sim = f(self.sim(z1,z2))
        between_sim=f(self.sim(z1,z2))

        return -torch.log(between_sim.diag()/(refl_sim.sum(1)+between_sim.sum(1)-refl_sim.diag()))
    
    def forward(self,z1:torch.Tensor,z2:torch.Tensor,mean:bool=True):
        l1=self.semi_loss(z1,z2)
        l2=self.semi_loss(z2,z1)
        ret=(l1+l2)*0.5
        ret=ret.mean() if mean else ret.sum()
        return ret