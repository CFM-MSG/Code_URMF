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
import torchvision


class ImageEncoder(nn.Module):
    def __init__(self, args):
        super(ImageEncoder, self).__init__()
        self.args = args
        model = torchvision.models.resnet152(pretrained=True)
        modules = list(model.children())[:-2]
        self.model = nn.Sequential(*modules)

        pool_func = (
            nn.AdaptiveAvgPool2d
            if args.img_embed_pool_type == "avg"
            else nn.AdaptiveMaxPool2d
        )

        if args.num_image_embeds in [1, 2, 3, 5, 7]:
            self.pool = pool_func((args.num_image_embeds, 1))
        elif args.num_image_embeds == 4:
            self.pool = pool_func((2, 2))
        elif args.num_image_embeds == 6:
            self.pool = pool_func((3, 2))
        elif args.num_image_embeds == 8:
            self.pool = pool_func((4, 2))
        elif args.num_image_embeds == 9:
            self.pool = pool_func((3, 3))

    def forward(self, x):
        # Bx3x224x224 -> Bx2048x7x7 -> Bx2048xN -> BxNx2048
        out = self.pool(self.model(x))
        out = torch.flatten(out, start_dim=2) #batchsize*2048*3
        out = out.transpose(1, 2).contiguous() #batchsize*3*2048
        return out  # BxNx2048

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

class ImageClf(nn.Module):
    def __init__(self, args):
        super(ImageClf, self).__init__()
        self.args = args
        self.img_encoder = ImageEncoder(args)
        self.clf=nn.Linear(128,args.n_classes)

        self.mu=nn.Sequential(       
        # nn.BatchNorm1d(args.img_hidden_sz* args.num_image_embeds, eps=2e-5, affine=False),
        # nn.Dropout(p=0.4),
        # Flatten(),
        nn.Linear(args.img_hidden_sz* args.num_image_embeds,128))
        # nn.BatchNorm1d(128,eps=2e-5))
        self.logvar=nn.Sequential(       
        # nn.BatchNorm1d(args.img_hidden_sz* args.num_image_embeds, eps=2e-5,affine=False),
        # nn.Dropout(p=0.4),
        # Flatten(),
        nn.Linear(args.img_hidden_sz* args.num_image_embeds, 128))
        # nn.BatchNorm1d(128,eps=2e-5))

    def forward(self, x):
        x = self.img_encoder(x)
        x = torch.flatten(x, start_dim=1) 
        mu=self.mu(x) #batch_size
        logvar=self.logvar(x) #batch_size
        x=self._reparameterize(mu,logvar)
        out=self.clf(x)
        return mu,logvar,out 
    
    def _reparameterize(self, mu, logvar):
        std = torch.exp(logvar).sqrt()
        epsilon = torch.randn_like(std)
        sampler = epsilon * std
        return mu + sampler