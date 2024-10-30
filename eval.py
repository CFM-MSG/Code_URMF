#!/usr/bin/env python3
#
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in   the
# LICENSE file in the root directory of this source tree.
#


import argparse
import os.path

from sklearn.metrics import f1_score, accuracy_score
from tqdm import tqdm

import json
import torch
import torch.nn as nn
import torch.optim as optim
from pytorch_pretrained_bert import BertAdam
import torch.nn.functional as F
from contrast_loss import Contrastive_loss
from sklearn.metrics import f1_score, accuracy_score
device = torch.device("cuda:0") 
torch.cuda.set_device(device)
from src.data.helpers import get_data_loaders
from src.models import get_model
from src.utils.logger import create_logger
from src.utils.utils import *

recoreds=[]

def get_args(parser):
    parser.add_argument("--batch_sz", type=int, default=16)
    parser.add_argument("--bert_model", type=str, default="./prebert")
    parser.add_argument("--data_path", type=str, default="./datasets/MVSA_Single/")
    parser.add_argument("--drop_img_percent", type=float, default=0.0)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--embed_sz", type=int, default=300)
    parser.add_argument("--freeze_img", type=int, default=3)
    parser.add_argument("--freeze_txt", type=int, default=5)
    parser.add_argument("--glove_path", type=str, default="datasets/glove_embeds/glove.840B.300d.txt")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=40)
    parser.add_argument("--hidden", nargs="*", type=int, default=[])
    parser.add_argument("--hidden_sz", type=int, default=768)
    parser.add_argument("--img_embed_pool_type", type=str, default="avg", choices=["max", "avg"])
    parser.add_argument("--img_hidden_sz", type=int, default=2048)
    parser.add_argument("--include_bn", type=int, default=True)
    parser.add_argument("--lr", type=float, default=5e-05)
    parser.add_argument("--lr_factor", type=float, default=0.5)
    parser.add_argument("--lr_patience", type=int, default=2)
    parser.add_argument("--max_epochs", type=int, default=100)
    parser.add_argument("--max_seq_len", type=int, default=512)
    parser.add_argument("--model", type=str, default="latefusion", choices=["bow", "img", "bert", "concatbow", "concatbert", "mmbt","latefusion"])
    parser.add_argument("--n_workers", type=int, default=12)
    parser.add_argument("--name", type=str, default="MVSA_Single_latefusion_model_run_df_1")
    parser.add_argument("--num_image_embeds", type=int, default=3)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--savedir", type=str, default="./saved/MVSA_Single")
    parser.add_argument("--seed", type=int, default=1701)
    parser.add_argument("--task", type=str, default="MVSA_Single", choices=["mmimdb", "vsnli", "food101","MVSA_Single"])
    parser.add_argument("--task_type", type=str, default="classification", choices=["multilabel", "classification"])
    parser.add_argument("--warmup", type=float, default=0.1)
    parser.add_argument("--weight_classes", type=int, default=1)
    parser.add_argument("--df", type=bool, default=True)
    parser.add_argument("--noise", type=float, default=5)
    parser.add_argument("--noise_type", type=str, default="Salt")

def get_criterion(args):
    if args.task_type == "multilabel":
        if args.weight_classes:
            freqs = [args.label_freqs[l] for l in args.labels]
            label_weights = (torch.FloatTensor(freqs) / args.train_data_len) ** -1
            criterion = nn.BCEWithLogitsLoss(pos_weight=label_weights.cuda())
        else:
            criterion = nn.BCEWithLogitsLoss()
    else:
        criterion = nn.CrossEntropyLoss()

    return criterion


def model_eval(i_epoch, data, model, args, criterion, store_preds=False):
    with torch.no_grad():
        losses, preds, tgts = [], [], []
        for batch in data:
            loss, out, tgt = model_forward(i_epoch, model, args, criterion, batch,mode='eval')
            losses.append(loss.item())

            if args.task_type == "multilabel":
                pred = torch.sigmoid(out).cpu().detach().numpy() > 0.5
            else:
                pred = torch.nn.functional.softmax(out, dim=1).argmax(dim=1).cpu().detach().numpy()

            preds.append(pred)
            tgt = tgt.cpu().detach().numpy()
            tgts.append(tgt)

    metrics = {"loss": np.mean(losses)}
    if args.task_type == "multilabel":
        tgts = np.vstack(tgts)
        preds = np.vstack(preds)
        metrics["macro_f1"] = f1_score(tgts, preds, average="macro")
        metrics["micro_f1"] = f1_score(tgts, preds, average="micro")
    else:
        tgts = [l for sl in tgts for l in sl]
        preds = [l for sl in preds for l in sl]
        metrics["acc"] = accuracy_score(tgts, preds)
        metrics["micro_f1"] = f1_score(tgts, preds, average="micro")


    if store_preds:
        store_preds_to_disk(tgts, preds, args)

    return metrics

def get_optimizer(model, args):
    if args.model in ["bert", "concatbert", "mmbt"]:
        total_steps = (
            args.train_data_len
            / args.batch_sz
            / args.gradient_accumulation_steps
            * args.max_epochs
        )
        param_optimizer = list(model.named_parameters())
        no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {"params": [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], "weight_decay": 0.01},
            {"params": [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], "weight_decay": 0.0,},
        ]
        optimizer = BertAdam(
            optimizer_grouped_parameters,
            lr=args.lr,
            warmup=args.warmup,
            t_total=total_steps,
        )
    else:
        optimizer = optim.Adam(model.parameters(), lr=args.lr)

    return optimizer


def get_scheduler(optimizer, args):
    return optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, "max", patience=args.lr_patience, verbose=True, factor=args.lr_factor
    )


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

def model_forward(i_epoch, model, args, criterion, batch,txt_history=None,img_history=None,mode='eval'):
    txt, segment, mask, img, tgt,_ = batch
    txt, img = txt.to(device), img.to(device)
    mask, segment = mask.to(device), segment.to(device)
    txt_img_logits, txt_logits, img_logits,txt_mu,txt_logvar,img_mu,img_logvar,mu,logvar,z=model(txt, mask, segment, img)


    tgt = tgt.to(device)

    conloss=con_loss(txt_mu,torch.exp(txt_logvar),img_mu,torch.exp(img_logvar))
    loss=totolloss(txt_img_logits, txt_logits,tgt,img_logits,txt_mu,txt_logvar,img_mu,img_logvar,mu,logvar,z)
    loss=loss+1e-5*KL_regular(txt_mu,txt_logvar,img_mu,img_logvar)+conloss*1e-3

    return loss,txt_img_logits,tgt


def train(args):

    set_seed(args.seed)
    args.savedir = os.path.join(args.savedir, args.name)
    os.makedirs(args.savedir, exist_ok=True)

    train_loader, val_loader, test_loaders = get_data_loaders(args)

    logger = create_logger("%s/eval_logfile.log" % args.savedir, args)

    model=torch.load(os.path.join(args.savedir, "model_best.pth"))
    model=model['state_dict']
    model.cuda()

    model.eval()


    accList=[]
    for test_name, test_loader in test_loaders.items():
        test_metrics = model_eval(
            np.inf, test_loader, model, args, None, store_preds=True
        )

        log_metrics(f"Test - {test_name}", test_metrics, args, logger) 
        accList.append(test_metrics['acc'])

    info = f"name:{args.name} seed:{args.seed} noise:{args.noise} test_acc: {accList[0]:0.5f}\n"


    result_json={
        'name':args.name,
        'method': args.model+'_df',
        'seed':args.seed,
        'noise':args.noise,
        'test_acc':accList[0],
    }

    path = f"eval_data/{args.task}_result_{args.noise_type}.json"
    if os.path.exists(path):
        with open(path, encoding='utf-8') as f:
            exist_json = json.load(f)
    else:
        exist_json = []

            
def cli_main():
    parser = argparse.ArgumentParser(description="Train Models")
    get_args(parser)
    args, remaining_args = parser.parse_known_args()
    assert remaining_args == [], remaining_args
    train(args)


if __name__ == "__main__":
    import warnings

    warnings.filterwarnings("ignore")

    cli_main()
