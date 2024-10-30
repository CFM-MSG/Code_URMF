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
from pytorch_pretrained_bert.modeling import BertModel


class BertEncoder(nn.Module):
    def __init__(self, args):
        super(BertEncoder, self).__init__()
        self.args = args
        self.bert = BertModel.from_pretrained(args.bert_model)

    def forward(self, txt, mask, segment):
        _, out = self.bert( 
            txt,
            token_type_ids=segment,
            attention_mask=mask,
            output_all_encoded_layers=False,
        )
        return out #16*768

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

class BertClf(nn.Module):
    def __init__(self, args):
        super(BertClf, self).__init__()
        self.args = args
        self.enc = BertEncoder(args) 
        self.clf = nn.Linear(128, args.n_classes)
        self.mu=nn.Sequential(      
            # nn.BatchNorm1d(args.hidden_sz, eps=2e-5,affine=False),
            # nn.Dropout(p=0.4),
            # Flatten(),
            nn.Linear(args.hidden_sz, 128))
            # nn.BatchNorm1d(128,eps=2e-5))
        self.logvar=nn.Sequential(       
        # nn.BatchNorm1d(args.hidden_sz, eps=2e-5,affine=False),
        # nn.Dropout(p=0.4),
        # Flatten(),
        nn.Linear(args.hidden_sz, 128))
        # nn.BatchNorm1d(128,eps=2e-5))
        self.clf.apply(self.enc.bert.init_bert_weights)
    
    def forward(self, txt, mask, segment):
        x = self.enc(txt, mask, segment) #x.shape=batch_size*768
        mu=self.mu(x) #batch_size*200
        logvar=self.logvar(x) #batch_size*200
        x=self._reparameterize(mu,logvar)
        out=self.clf(x)
        return mu,logvar,out 
    
    def _reparameterize(self, mu, logvar):
        std = torch.exp(logvar).sqrt()
        epsilon = torch.randn_like(std)
        sampler = epsilon * std
        return mu + sampler

        
