#!/usr/bin/env python3
#
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import torch
import torch.nn as nn

from attention import ChannelGate
from models.bert import BertEncoder,BertClf
from models.image import ImageEncoder,ImageClf
from src.models.tool import cog_uncertainty_normal, cog_uncertainty_sample



def reparameterise(mu, std):
    """
    mu : [batch_size,z_dim]
    std : [batch_size,z_dim]        
    """        
    # get epsilon from standard normal
    eps = torch.randn_like(std)
    return mu + std*eps


class MultimodalLateFusionClf(nn.Module):
    def __init__(self, args):
        super(MultimodalLateFusionClf, self).__init__()
        self.args = args

        self.fusion=ChannelGate(3,3,'avg')
        self.txtclf = BertClf(args)
        self.imgclf= ImageClf(args)
        self.mu=nn.Linear(128,128)
        self.logvar=nn.Linear(128,128)
        self.IB_classfier=nn.Linear(128,3)
        self.fc_fusion1=nn.Sequential(nn.Linear(128,3))      
        

    def forward(self, txt, mask, segment, img):
        txt_mu,txt_logvar,txt_out = self.txtclf(txt, mask, segment) 
        img_mu,img_logvar,img_out = self.imgclf(img)

        txt_var=torch.exp(txt_logvar)
        img_var=torch.exp(img_logvar)

        def get_supp_mod(key):
            if key == "l":
                return img_mu
            elif key == "v":
                return txt_mu
            else:
                raise KeyError


        l_sample, v_sample = cog_uncertainty_sample(txt_mu, txt_var, img_mu, img_logvar, sample_times=10)  
        sample_dict = {
            "l": l_sample, 
            "v": v_sample
        }
        cog_uncertainty_dict = {}
        with torch.no_grad():
            for key, sample_tensor in sample_dict.items():
                bsz, sample_times, dim = sample_tensor.shape
                sample_tensor = sample_tensor.reshape(bsz * sample_times, dim)
                sample_tensor = sample_tensor.unsqueeze(1)  
                supp_mod = get_supp_mod(key)
                supp_mod = supp_mod.unsqueeze(1)
                supp_mod = supp_mod.unsqueeze(1).repeat(1, sample_times, 1, 1)
                supp_mod = supp_mod.reshape(bsz * sample_times, 1, dim)  
                feature = torch.cat([supp_mod, sample_tensor], dim=1)
    
                feature_fusion=self.fusion(feature)
                mu=self.mu(feature_fusion)
                logvar=self.logvar(feature_fusion)
                z=reparameterise(mu,torch.exp(logvar))
                z=self.IB_classfier(z)
                txt_img_out=self.fc_fusion1(mu)
                
                cog_un = torch.var(txt_img_out, dim=-1)  
                cog_uncertainty_dict[key] = cog_un
            
        cog_uncertainty_dict = cog_uncertainty_normal(cog_uncertainty_dict)


        weight=torch.softmax(torch.stack([img_var, txt_var]), dim=0)
        img_w=weight[1]
        txt_w=weight[0]

        feature_txt=txt_mu*txt_w
        feature_img=img_mu*img_w


        feature=torch.stack((feature_txt,feature_img),dim=1)
        feature_fusion=self.fusion(feature)
        mu=self.mu(feature_fusion)
        logvar=self.logvar(feature_fusion)
        z=reparameterise(mu,torch.exp(logvar))
        z=self.IB_classfier(z)
        txt_img_out=self.fc_fusion1(mu)



        return [txt_img_out,txt_out,img_out,txt_mu,txt_logvar,img_mu,img_logvar,mu,logvar,z,cog_uncertainty_dict] 