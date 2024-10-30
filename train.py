#!/usr/bin/env python3
#
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#


import argparse
from sklearn.metrics import f1_score, accuracy_score
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from pytorch_pretrained_bert import BertAdam
import torch.nn.functional as F

device = torch.device("cuda:0") 
torch.cuda.set_device(device)

from util import Contrastive_loss,totolloss,con_loss,KL_regular
from src.data.helpers import get_data_loaders
from src.models import get_model
from src.utils.logger import create_logger
from src.utils.utils import *

import time

def get_args(parser):
    parser.add_argument("--batch_sz", type=int, default=8)
    parser.add_argument("--bert_model", type=str, default="./prebert")
    parser.add_argument("--data_path", type=str, default="./datasets/MVSA_Single/")
    parser.add_argument("--drop_img_percent", type=float, default=0.0)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--embed_sz", type=int, default=300)
    parser.add_argument("--freeze_img", type=int, default=3)
    parser.add_argument("--freeze_txt", type=int, default=5)
    parser.add_argument("--glove_path", type=str, default="./datasets/glove_embeds/glove.840B.300d.txt")
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
    parser.add_argument("--n_workers", type=int, default=4)
    parser.add_argument("--name", type=str, default="URMF")
    parser.add_argument("--num_image_embeds", type=int, default=3)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--savedir", type=str, default="./saved/MVSA_Single")
    parser.add_argument("--seed", type=int, default=1699)
    parser.add_argument("--task", type=str, default="MVSA_Single", choices=["MVSA_Single"])
    parser.add_argument("--task_type", type=str, default="classification", choices=["multilabel", "classification"])
    parser.add_argument("--warmup", type=float, default=0.1)
    parser.add_argument("--weight_classes", type=int, default=1)
    parser.add_argument("--df", type=bool, default=True)
    parser.add_argument("--noise", type=float, default=0.0)
    parser.add_argument("--log_marker", type=str, default='')
    parser.add_argument("--modulation_starts", type=int, default=5)
    parser.add_argument("--modulation_ends", type=int, default=100)
    parser.add_argument("--zeta", type=float, default=0.01)



def get_criterion(args):

    criterion = nn.CrossEntropyLoss()

    return criterion


def get_optimizer(model, args):

    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    return optimizer


def get_scheduler(optimizer, args):
    return optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, "max", patience=args.lr_patience, verbose=True, factor=args.lr_factor
    )


def model_eval(i_epoch, data, model, args, criterion, store_preds=False):
    with torch.no_grad():
        losses, preds, tgts = [], [], []
        for batch in data:
            loss, out, tgt, cog_un = model_forward(i_epoch, model, args, criterion, batch,mode='eval')
            losses.append(loss.item())

            if args.task_type == "multilabel":
                pred = torch.sigmoid(out).cpu().detach().numpy() > 0.5
            else:
                pred = torch.nn.functional.softmax(out, dim=1).argmax(dim=1).cpu().detach().numpy()

            preds.append(pred)
            tgt = tgt.cpu().detach().numpy()
            tgts.append(tgt)

    metrics = {"loss": np.mean(losses)}

    tgts = [l for sl in tgts for l in sl]
    preds = [l for sl in preds for l in sl]
    metrics["acc"] = accuracy_score(tgts, preds)
    metrics["micro_f1"] = f1_score(tgts, preds, average="weighted")


    if store_preds:
        store_preds_to_disk(tgts, preds, args)

    return metrics


def model_forward(i_epoch, model, args, criterion, batch,txt_history=None,img_history=None,mode='eval'):
    txt, segment, mask, img, tgt,idx = batch
    # print(txt)
    # print(img)
    freeze_img = i_epoch < args.freeze_img
    freeze_txt = i_epoch < args.freeze_txt

    txt, img = txt.to(device), img.to(device)
    mask, segment = mask.to(device), segment.to(device)
    # out = model(txt, mask, segment, img)
    txt_img_logits, txt_logits, img_logits,txt_mu,txt_logvar,img_mu,img_logvar,mu,logvar,z, cog_un=model(txt, mask, segment, img)


    tgt = tgt.to(device)
    # loss = criterion(out, tgt)

    conloss=con_loss(txt_mu,torch.exp(txt_logvar),img_mu,torch.exp(img_logvar))
    loss=totolloss(txt_img_logits, txt_logits,tgt,img_logits,txt_mu,txt_logvar,img_mu,img_logvar,mu,logvar,z)
    loss=loss+1e-5*KL_regular(txt_mu,txt_logvar,img_mu,img_logvar)+conloss*1e-3
    # loss=loss+0.01*conloss

    return loss,txt_img_logits,tgt, cog_un

def train(args):

    set_seed(args.seed)
    args.savedir = os.path.join(args.savedir, args.name)
    os.makedirs(args.savedir, exist_ok=True)

    train_loader, val_loader, test_loaders = get_data_loaders(args)

    # print('adsasd')
    model = get_model(args)
    # print('1341232')
    criterion = get_criterion(args)
    optimizer = get_optimizer(model, args)
    scheduler = get_scheduler(optimizer, args)


    current_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
    dataset = "MVSA"
    if not os.path.exists(f"./log/{dataset}"):
        os.makedirs(f"./log/{dataset}")
    if args.log_marker != "":
        log_name = str(current_time)+"_"+args.log_marker
    else:
        log_name = str(current_time)
    log_path = os.path.join(f"./log/{dataset}", f"{log_name}.log")

    logger = create_logger(log_path, args)

    argsDict = args.__dict__
    for eachArg, value in argsDict.items():
        logger.info('{}:{}'.format(eachArg, value))
    logger.info("==============================================")

    # logger.info(model)
    model.cuda()


    torch.save(args, os.path.join(args.savedir, "args.pt"))

    start_epoch, global_step, n_no_improve, best_metric = 0, 0, 0, -np.inf

    if os.path.exists(os.path.join(args.savedir, "checkpoint.pt")):
        checkpoint = torch.load(os.path.join(args.savedir, "checkpoint.pt"))
        start_epoch = checkpoint["epoch"]
        n_no_improve = checkpoint["n_no_improve"]
        best_metric = checkpoint["best_metric"]
        model.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])

    logger.info("Training..")
    txt_history = History(len(train_loader.dataset))
    img_history = History(len(train_loader.dataset))

    for i_epoch in range(start_epoch, args.max_epochs):
        train_losses = []
        model.train()
        optimizer.zero_grad()
        preds, tgts = [], []
        for batch in tqdm(train_loader,total=len(train_loader)):
        # for batch in train_loader:
            loss, out, tgt, cog_uncertainty_dict = model_forward(i_epoch, model, args, criterion, batch,txt_history,img_history,mode='train')
            train_losses.append(loss.item())



            loss.backward()

            if args.modulation_starts <= i_epoch <= args.modulation_ends: # bug fixed
                coeff_l = args.zeta * cog_uncertainty_dict['l'].mean()
                coeff_v = args.zeta * cog_uncertainty_dict['v'].mean()
                for name, parms in model.named_parameters():
                    if parms.grad == None:
                        continue
                    if any( _ in name for _ in ["txtclf"]):
                        parms.grad = parms.grad * (1+coeff_v)
                    if any( _ in name for _ in ["imgclf"]):
                        parms.grad = parms.grad * (1+coeff_l)
            else:
                pass
            
            
            global_step += 1
            if global_step % args.gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

            pred = torch.nn.functional.softmax(out, dim=1).argmax(dim=1).cpu().detach().numpy()

            preds.append(pred)
            tgt = tgt.cpu().detach().numpy()
            tgts.append(tgt)

        tgts = [l for sl in tgts for l in sl]
        preds = [l for sl in preds for l in sl]
        train_acc = accuracy_score(tgts, preds)


        model.eval()
        metrics = model_eval(i_epoch, val_loader, model, args, criterion)
        logger.info("Train Loss: {:.5f}  Acc: {:.5f}".format(np.mean(train_losses),train_acc))
        log_metrics("Val", metrics, args, logger)

        tuning_metric = (
            metrics["micro_f1"] if args.task_type == "multilabel" else metrics["acc"]
        )
        scheduler.step(tuning_metric)
        is_improvement = tuning_metric > best_metric
        if is_improvement:
            best_metric = tuning_metric
            n_no_improve = 0
        else:
            n_no_improve += 1
 
        save_checkpoint(
            {
                "epoch": i_epoch + 1,
                "state_dict": model,
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "n_no_improve": n_no_improve,
                "best_metric": best_metric,
            },
            is_improvement,
            args.savedir,
        )

        model.eval()
        for test_name, test_loader in test_loaders.items():
            test_metrics = model_eval(
                np.inf, test_loader, model, args, criterion, store_preds=True
            )
            log_metrics(f"Test - {test_name}", test_metrics, args, logger)


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
