import torch
import torch.nn as nn
import torch.nn.functional as F
from contrast_loss import Contrastive_loss
import numpy as np
import pdb

def KL_regular(mu_1, var_1, mu_2, var_2, mu_3, var_3):

    KL_loss_1=var_2.log()-var_1.log()+((var_1.pow(2)+(mu_1-mu_2).pow(2))/(2*var_2.pow(2)))-0.5
    KL_loss_1=KL_loss_1.sum(dim=1).mean()

    KL_loss_2=var_3.log()-var_1.log()+((var_1.pow(2)+(mu_1-mu_3).pow(2))/(2*var_3.pow(2)))-0.5
    KL_loss_2=KL_loss_2.sum(dim=1).mean()

    sub_kl_loss_1 = -(1 + var_1.log() - mu_1.pow(2) - var_1) / 2
    sub_kl_loss_1 = sub_kl_loss_1.sum(dim=1).mean()

    sub_kl_loss_2 = -(1 + var_2.log() - mu_2.pow(2) - var_2) / 2
    sub_kl_loss_2 = sub_kl_loss_2.sum(dim=1).mean()

    sub_kl_loss_3 = -(1 + var_3.log() - mu_3.pow(2) - var_3) / 2
    sub_kl_loss_3 = sub_kl_loss_3.sum(dim=1).mean()

    sub_kl_loss=sub_kl_loss_1 + sub_kl_loss_2 + sub_kl_loss_3

    return KL_loss_1 + KL_loss_2 + sub_kl_loss*1e3

def reparameterise(mu, std):
    """
    mu : [batch_size,z_dim]
    std : [batch_size,z_dim]        
    """        
    # get epsilon from standard normal
    eps = torch.randn_like(std)
    return mu + std*eps

def con_loss(txt_mu, txt_logvar, img_mu, img_logvar, aou_mu, aou_logvar):
    Conloss=Contrastive_loss(0.5)


    while True:
        t_z1 = reparameterise(txt_mu, txt_logvar)
        t_z2 = reparameterise(txt_mu, txt_logvar)
        
        if not np.array_equal(t_z1, t_z2):
            break 


    while True:
        i_z1=reparameterise(img_mu,img_logvar)
        i_z2=reparameterise(img_mu,img_logvar)
    
        if not np.array_equal(i_z1, i_z2):
            break 



    while True:
        a_z1=reparameterise(aou_mu,aou_logvar)
        a_z2=reparameterise(aou_mu,aou_logvar)
    
        if not np.array_equal(a_z1, a_z2):
            break 

    loss_t=Conloss(t_z1,t_z2)
    loss_i=Conloss(i_z1,i_z2)
    loss_a=Conloss(a_z1,a_z2)
    
    return loss_t + loss_i + loss_a


def cog_uncertainty_sample(mu_l, var_l, mu_v, var_v, sample_times=10):

    l_list = []
    for _ in range(sample_times):
        l_list.append(reparameterise(mu_l, var_l))
    l_sample = torch.stack(l_list, dim=1)

    v_list = []
    for _ in range(sample_times):
        v_list.append(reparameterise(mu_v, var_v))
    v_sample = torch.stack(v_list, dim=1)
    
    return l_sample, v_sample


def cog_uncertainty_normal(unc_dict, normal_type="None"):

    key_list = [k for k, _ in unc_dict.items()]
    comb_list = [t for _, t in unc_dict.items()]
    comb_t = torch.stack(comb_list, dim=1)
    mat = torch.exp(torch.reciprocal(comb_t))
    mat_sum = mat.sum(dim=-1, keepdim=True)
    weight = mat / mat_sum

    if normal_type == "minmax":
        weight = weight / torch.max(weight, dim=1)[0].unsqueeze(-1)  # [bsz, mod_num]
        for i, key in enumerate(key_list):
            unc_dict[key] = weight[:, i]
    else:
        pass
        # raise TypeError("Unsupported Operations at cog_uncertainty_normal!")

    return unc_dict
