import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
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

def totolloss(txt_img_logits, txt_logits,tgt,img_logits,txt_mu,txt_logvar,img_mu,img_logvar,mu,logvar,z):

    txt_kl_loss = -(1 + txt_logvar - txt_mu.pow(2) - txt_logvar.exp()) / 2  
    txt_kl_loss = txt_kl_loss.sum(dim=1).mean()

    img_kl_loss = -(1 + img_logvar - img_mu.pow(2) - img_logvar.exp()) / 2  
    img_kl_loss = img_kl_loss.sum(dim=1).mean()

    kl_loss = -(1 + logvar - mu.pow(2) - logvar.exp()) / 2  
    kl_loss = kl_loss.sum(dim=1).mean()
    IB_loss=F.cross_entropy(z,tgt)

    fusion_cls_loss=F.cross_entropy(txt_img_logits,tgt)

    totol_loss=fusion_cls_loss+1e-3*kl_loss+1e-3*txt_kl_loss+1e-3*img_kl_loss+1e-3*IB_loss
    return totol_loss

def KL_regular(mu_1,logvar_1,mu_2,logvar_2):
    var_1=torch.exp(logvar_1)
    var_2=torch.exp(logvar_2)
    KL_loss=logvar_2-logvar_1+((var_1.pow(2)+(mu_1-mu_2).pow(2))/(2*var_2.pow(2)))-0.5
    KL_loss=KL_loss.sum(dim=1).mean()
    return KL_loss

def reparameterise(mu, std):
    """
    mu : [batch_size,z_dim]
    std : [batch_size,z_dim]        
    """        
    # get epsilon from standard normal
    eps = torch.randn_like(std)
    return mu + std*eps

def con_loss(txt_mu,txt_logvar,img_mu,img_logvar):
    Conloss=Contrastive_loss(0.5)
    while True:
        t_z1 = reparameterise(txt_mu, txt_logvar)
        t_z2 = reparameterise(txt_mu, txt_logvar)
        
        if not np.array_equal(t_z1, t_z2):
            break 
    while True:
        i_z1=reparameterise(img_mu,img_logvar)
        i_z2=reparameterise(img_mu,img_logvar)
        
        if not np.array_equal(t_z1, t_z2):
            break 


    loss_t=Conloss(t_z1,t_z2)
    loss_i=Conloss(i_z1,i_z2)
    
    return loss_t+loss_i