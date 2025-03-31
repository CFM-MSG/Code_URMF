from __future__ import absolute_import, division, print_function

import pdb
import argparse
import csv
import os
import random
import pickle
import sys
import numpy as np
import torch.optim as optim
from itertools import chain
from torch.optim import AdamW, Adam
from typing import *
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score, f1_score

from transformers import T5ForConditionalGeneration, T5Tokenizer, AdamW,T5Config

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange

from torch.nn import CrossEntropyLoss, L1Loss, MSELoss
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import matthews_corrcoef
from transformers import BertTokenizer, XLNetTokenizer, get_linear_schedule_with_warmup
from transformers.optimization import AdamW
#from cmib import MIB   #######################choose the MIB variants here
#from emib import MIB
#from lmib import MIB

from argparse_utils import str2bool, seed
from global_configs import ACOUSTIC_DIM, VISUAL_DIM, DEVICE


import logging
import time

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default="/datasets/MOSI/MOSI.pkl")
parser.add_argument("--max_seq_length", type=int, default=50)
parser.add_argument("--train_batch_size", type=int, default=16)
parser.add_argument("--dev_batch_size", type=int, default=128)
parser.add_argument("--test_batch_size", type=int, default=128)
parser.add_argument("--n_epochs", type=int, default=200)
parser.add_argument("--beta_shift", type=float, default=1.0)
parser.add_argument("--dropout_prob", type=float, default=0.5)
parser.add_argument(
    "--model",
    type=str,
    choices=["bert-base-uncased"],
    default="bert-base-uncased",
)
parser.add_argument("--learning_rate", type=float, default=1e-5)
parser.add_argument("--gradient_accumulation_step", type=int, default=1)
parser.add_argument("--warmup_proportion", type=float, default=0.1)
parser.add_argument("--seed", type=int, default=37) #5576
parser.add_argument("--mib", type=str, default='emib')
parser.add_argument("--logger", action='store_true', default=False)
parser.add_argument("--log_marker", type=str, default='')
parser.add_argument("--modulation_starts", type=int, default=10)
parser.add_argument("--modulation_ends", type=int, default=200)
parser.add_argument("--zeta", type=float, default=1.0)




args = parser.parse_args()



if args.mib == 'emib':
    from emib import MIB
    print('selecting emib!')


current_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
dataset = str(args.dataset).split("/")[-2]
if not os.path.exists(f"./log/{dataset}"):
    os.makedirs(f"./log/{dataset}")
if args.log_marker != "":
    log_name = str(current_time)+"_"+args.log_marker
else:
    log_name = str(current_time)
log_path = os.path.join(f"./log/{dataset}", f"{log_name}.log")

# logging.basicConfig(filename=log_path, level=logging.DEBUG, filemode = 'w', format='%(levelname)s:%(asctime)s:%(message)s', datefmt='%Y-%d-%m %H:%M:%S')

logger = logging.getLogger(log_path)

logger.setLevel(level=logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s')

stream_handler = logging.StreamHandler()
stream_handler.setLevel(logging.DEBUG)
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)

if args.logger:
    file_handler = logging.FileHandler(log_path)
    file_handler.setLevel(level=logging.INFO)
    file_handler.setFormatter(formatter)
    logger.info('Logging into {}'.format(log_path))
    logger.addHandler(file_handler)



argsDict = args.__dict__
for eachArg, value in argsDict.items():
    logger.info('{}:{}'.format(eachArg, value))
logger.info("==============================================")



def return_unk():
    return 0


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, visual, acoustic, input_mask, segment_ids, label_id):
        self.input_ids = input_ids
        self.visual = visual
        self.acoustic = acoustic
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id


class MultimodalConfig(object):
    def __init__(self, beta_shift, dropout_prob):
        self.beta_shift = beta_shift
        self.dropout_prob = dropout_prob


def convert_to_features(examples, max_seq_length, tokenizer):
    features = []

    for (ex_index, example) in enumerate(examples):

        (words, visual, acoustic), label_id, segment = example
       # print(words)
        tokens, inversions = [], []
        for idx, word in enumerate(words):
            tokenized = tokenizer.tokenize(word)
           # print(tokenized)
            tokens.extend(tokenized)
            inversions.extend([idx] * len(tokenized))

        # Check inversion
        assert len(tokens) == len(inversions)

        aligned_visual = []
        aligned_audio = []

        for inv_idx in inversions:
            aligned_visual.append(visual[inv_idx, :])
            aligned_audio.append(acoustic[inv_idx, :])

        visual = np.array(aligned_visual)
        acoustic = np.array(aligned_audio)

        # Truncate input if necessary
        if len(tokens) > max_seq_length - 2:
            tokens = tokens[: max_seq_length - 2]
            acoustic = acoustic[: max_seq_length - 2]
            visual = visual[: max_seq_length - 2]

        if args.model == "bert-base-uncased":
            prepare_input = prepare_bert_input


        input_ids, visual, acoustic, input_mask, segment_ids = prepare_input(
            tokens, visual, acoustic, tokenizer
        )

        # Check input length
        assert len(input_ids) == args.max_seq_length
        assert len(input_mask) == args.max_seq_length
        assert len(segment_ids) == args.max_seq_length
        assert acoustic.shape[0] == args.max_seq_length
        assert visual.shape[0] == args.max_seq_length

        features.append(
            InputFeatures(
                input_ids=input_ids,
                input_mask=input_mask,
                segment_ids=segment_ids,
                visual=visual,
                acoustic=acoustic,
                label_id=label_id,
            )
        )
    return features


def prepare_bert_input(tokens, visual, acoustic, tokenizer):
    CLS = tokenizer.cls_token
    SEP = tokenizer.sep_token
    tokens = [CLS] + tokens + [SEP]

    # Pad zero vectors for acoustic / visual vectors to account for [CLS] / [SEP] tokens
    acoustic_zero = np.zeros((1, ACOUSTIC_DIM))
    acoustic = np.concatenate((acoustic_zero, acoustic, acoustic_zero))
    visual_zero = np.zeros((1, VISUAL_DIM))
    visual = np.concatenate((visual_zero, visual, visual_zero))

    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    segment_ids = [0] * len(input_ids)
    input_mask = [1] * len(input_ids)

    pad_length = args.max_seq_length - len(input_ids)

    acoustic_padding = np.zeros((pad_length, ACOUSTIC_DIM))
    acoustic = np.concatenate((acoustic, acoustic_padding))

    visual_padding = np.zeros((pad_length, VISUAL_DIM))
    visual = np.concatenate((visual, visual_padding))

    padding = [0] * pad_length

    # Pad inputs
    input_ids += padding
    input_mask += padding
    segment_ids += padding

    return input_ids, visual, acoustic, input_mask, segment_ids




def get_tokenizer(model):
    if model == "bert-base-uncased":
        return BertTokenizer.from_pretrained('./prebert')
    
    else:
        raise ValueError(
            "Expected 'bert-base-uncased' or 'xlnet-base-cased, but received {}".format(
                model
            )
        )


def get_appropriate_dataset(data):

    tokenizer = get_tokenizer(args.model)

    features = convert_to_features(data, args.max_seq_length, tokenizer)
    all_input_ids = torch.tensor(
        [f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor(
        [f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor(
        [f.segment_ids for f in features], dtype=torch.long)
    all_visual = torch.tensor([f.visual for f in features], dtype=torch.float)
    all_acoustic = torch.tensor(
        [f.acoustic for f in features], dtype=torch.float)
    all_label_ids = torch.tensor(
        [f.label_id for f in features], dtype=torch.float)

    dataset = TensorDataset(
        all_input_ids,
        all_visual,
        all_acoustic,
        all_input_mask,
        all_segment_ids,
        all_label_ids,
    )
    return dataset


def set_up_data_loader():
    with open(args.dataset, "rb") as handle:
        data = pickle.load(handle)

    train_data = data["train"]
    dev_data = data["dev"]
    test_data = data["test"]

    train_dataset = get_appropriate_dataset(train_data)
    dev_dataset = get_appropriate_dataset(dev_data)
    test_dataset = get_appropriate_dataset(test_data)

    num_train_optimization_steps = (
        int(
            len(train_dataset) / args.train_batch_size /
            args.gradient_accumulation_step
        )
        * args.n_epochs
    )

    train_dataloader = DataLoader(
        train_dataset, batch_size=args.train_batch_size, shuffle=True,drop_last=True
    )

    dev_dataloader = DataLoader(
        dev_dataset, batch_size=args.dev_batch_size, shuffle=True
    )

    test_dataloader = DataLoader(
        test_dataset, batch_size=args.test_batch_size, shuffle=True,
    )

    return (
        train_dataloader,
        dev_dataloader,
        test_dataloader,
        num_train_optimization_steps,
    )


def set_random_seed(seed: int):
    """
    Helper function to seed experiment for reproducibility.
    If -1 is provided as seed, experiment uses random seed from 0~9999

    Args:
        seed (int): integer to be used as seed, use -1 to randomly seed experiment
    """
 
    print("Seed: {}".format(seed))

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.deterministic = True

    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def prep_for_training(num_train_optimization_steps: int):
    multimodal_config = MultimodalConfig(
        beta_shift=args.beta_shift, dropout_prob=args.dropout_prob
    )

    if args.model == "bert-base-uncased":
        model = MIB(multimodal_config=multimodal_config, num_labels=1)
   

    total_para = 0
    # for param in model.parameters():
    #     total_para += np.prod(param.size())
    # print('total parameter for the model: ', total_para)



    model.to(DEVICE)

    return model



def train_epoch(model: nn.Module, train_dataloader: DataLoader, epoch=0):
    model.train()
    tr_loss = 0
    preds = []
    labels = []
    nb_tr_examples, nb_tr_steps = 0, 0

    optimizer_l = Adam(model.bert.parameters(), lr=model.llr)
    optimizer_all = getattr(optim, 'Adam')(chain(model.trans_a.parameters(), model.trans_v.parameters(), model.fusion.parameters(), model.proj_a.parameters(), model.proj_v.parameters()), lr=model.lr)

    scheduler_all=optim.lr_scheduler.ReduceLROnPlateau(
        optimizer_all, "min", patience=2, verbose=True, factor=0.9
    )
    
    scheduler_l=optim.lr_scheduler.ReduceLROnPlateau(
        optimizer_l, "min", patience=2, verbose=True, factor=0.9
    )
    
    for step, batch in enumerate(train_dataloader):
        batch = tuple(t.to(DEVICE) for t in batch)
        input_ids, visual, acoustic, input_mask, segment_ids, label_ids = batch
        visual = torch.squeeze(visual, 1)
        acoustic = torch.squeeze(acoustic, 1)
        outputs, cog_uncertainty_dict, loss_fusion, loss_task = model( 
            input_ids,
            visual,
            acoustic,
            label_ids,
            token_type_ids=segment_ids,
            attention_mask=input_mask,
            labels=None,
        )

        logits = outputs #+ outputa + outputv


        loss_fct = MSELoss()
        mse_loss_task = loss_fct(logits.view(-1), label_ids.view(-1))

        loss = loss_fusion + loss_task + mse_loss_task

        loss_task += mse_loss_task

        optimizer_l.zero_grad()
        optimizer_all.zero_grad()

        # Fusion Loss
        loss_fusion.backward(retain_graph = True)
        optimizer_all.step()
        optimizer_l.step()

        # Task-specific Gradient Modulation
        loss_task.backward()
        if args.modulation_starts <= epoch <= args.modulation_ends: # bug fixed
            coeff_l = args.zeta * cog_uncertainty_dict['l'].mean()
            coeff_a = args.zeta * cog_uncertainty_dict['a'].mean()
            coeff_v = args.zeta * cog_uncertainty_dict['v'].mean()
            for name, parms in model.named_parameters():
                # layer = str(name).split('.')
                if parms.grad == None:
                    continue
                if any( _ in name for _ in ["trans_a", "proj_a", "mu_a","log_var_a"]):
                    # parms.grad = parms.grad * (1+coeff_a) + torch.zeros_like(parms.grad).normal_(0, parms.grad.std().item() + 1e-8)
                    parms.grad = parms.grad * (1+coeff_a)
                if any( _ in name for _ in ["trans_v", "proj_v", "mu_v","log_var_v"]):
                    # parms.grad = parms.grad * (1+coeff_v) + torch.zeros_like(parms.grad).normal_(0, parms.grad.std().item() + 1e-8)
                    parms.grad = parms.grad * (1+coeff_v)
                if any( _ in name for _ in ["bert", "mu_l","log_var_l"]):
                    # parms.grad = parms.grad * (1+coeff_l) + torch.zeros_like(parms.grad).normal_(0, parms.grad.std().item() + 1e-8)
                    parms.grad = parms.grad * (1+coeff_l)
        else:
            pass

        optimizer_all.step()
        optimizer_l.step()

        tr_loss += loss.item()
        nb_tr_steps += 1

        logits = logits.detach().cpu().numpy()
        label_ids = label_ids.detach().cpu().numpy()

        logits = np.squeeze(logits).tolist()
        label_ids = np.squeeze(label_ids).tolist()

        preds.extend(logits)
        labels.extend(label_ids)

    preds = np.array(preds)
    labels = np.array(labels)       

    return tr_loss / nb_tr_steps,scheduler_all,scheduler_l,preds,labels


def eval_epoch(model: nn.Module, dev_dataloader: DataLoader):
    model.eval()
    dev_loss = 0
    nb_dev_examples, nb_dev_steps = 0, 0
    preds = []
    labels = []
    with torch.no_grad():
        # for step, batch in enumerate(tqdm(dev_dataloader, desc="Iteration")):
        for step, batch in enumerate(dev_dataloader):
            batch = tuple(t.to(DEVICE) for t in batch)

            input_ids, visual, acoustic, input_mask, segment_ids, label_ids = batch
            visual = torch.squeeze(visual, 1)
            acoustic = torch.squeeze(acoustic, 1)
            outputs = model.test(
                input_ids,
                 visual,
                 acoustic,
                # label_ids,
                token_type_ids=segment_ids,
                attention_mask=input_mask,
               # labels=None,
            )

            logits = outputs

            loss_fct = MSELoss()
            loss = loss_fct(logits.view(-1), label_ids.view(-1))

            if args.gradient_accumulation_step > 1:
                loss = loss / args.gradient_accumulation_step

            dev_loss += loss.item()
            nb_dev_steps += 1

            logits = outputs

            logits = logits.detach().cpu().numpy()
            label_ids = label_ids.detach().cpu().numpy()

            logits = np.squeeze(logits).tolist()
            label_ids = np.squeeze(label_ids).tolist()

            preds.extend(logits)
            labels.extend(label_ids)

        preds = np.array(preds)
        labels = np.array(labels)

    return dev_loss / nb_dev_steps,preds,labels


def test_epoch(model: nn.Module, test_dataloader: DataLoader):
    model.eval()
    preds = []
    labels = []

    with torch.no_grad():
        # for batch in tqdm(test_dataloader):
        for batch in test_dataloader:
            batch = tuple(t.to(DEVICE) for t in batch)

            input_ids, visual, acoustic, input_mask, segment_ids, label_ids = batch
            visual = torch.squeeze(visual, 1)
            acoustic = torch.squeeze(acoustic, 1)
            outputs = model.test(
                input_ids,
                 visual,
                 acoustic,
                token_type_ids=segment_ids,
                attention_mask=input_mask,
                labels=None,
            )

            logits = outputs

            logits = logits.detach().cpu().numpy()
            label_ids = label_ids.detach().cpu().numpy()

            logits = np.squeeze(logits).tolist()
            label_ids = np.squeeze(label_ids).tolist()

            preds.extend(logits)
            labels.extend(label_ids)

        preds = np.array(preds)
        labels = np.array(labels)

    return preds, labels

def multiclass_acc(preds, truths):
    """
    Compute the multiclass accuracy w.r.t. groundtruth

    :param preds: Float array representing the predictions, dimension (N,)
    :param truths: Float/int array representing the groundtruth classes, dimension (N,)
    :return: Classification accuracy
    """
    return np.sum(np.round(preds) == np.round(truths)) / float(len(truths))

def test_score_model(model: nn.Module, test_dataloader: DataLoader, use_zero=False):

    preds, y_test = test_epoch(model, test_dataloader)
    non_zeros = np.array(
        [i for i, e in enumerate(y_test) if e != 0 or use_zero])

    test_preds_a7 = np.clip(preds, a_min=-3., a_max=3.)
    test_truth_a7 = np.clip(y_test, a_min=-3., a_max=3.)
    mult_a7 = multiclass_acc(test_preds_a7, test_truth_a7)


    preds = preds[non_zeros]
    y_test = y_test[non_zeros]



    mae = np.mean(np.absolute(preds - y_test))
    corr = np.corrcoef(preds, y_test)[0][1]

    preds = preds >= 0
    y_test = y_test >= 0

    f_score = f1_score(y_test, preds, average="weighted")
    acc = accuracy_score(y_test, preds)


    return acc, mae, corr, f_score, mult_a7


def compute(preds,y_test,use_zero=False):
    non_zeros = np.array(
    [i for i, e in enumerate(y_test) if e != 0 or use_zero])

    test_preds_a7 = np.clip(preds, a_min=-3., a_max=3.)
    test_truth_a7 = np.clip(y_test, a_min=-3., a_max=3.)
    mult_a7 = multiclass_acc(test_preds_a7, test_truth_a7)


    preds = preds[non_zeros]
    y_test = y_test[non_zeros]



    mae = np.mean(np.absolute(preds - y_test))
    corr = np.corrcoef(preds, y_test)[0][1]

    preds = preds >= 0
    y_test = y_test >= 0

    f_score = f1_score(y_test, preds, average="weighted")
    acc = accuracy_score(y_test, preds)


    return acc, mae, corr, f_score, mult_a7

def train(
    model,
    train_dataloader,
    validation_dataloader,
    test_data_loader
):
    valid_losses = []
    test_accuracies = []
    best_acc7_f = 0
    best_epoch = 0
    for epoch_i in range(int(args.n_epochs)):
        train_loss,scheduler_all,scheduler_l,train_preds,train_labels = train_epoch(model, train_dataloader,epoch=epoch_i)
        train_acc, train_mae, train_corr, train_f_score, train_acc7=compute(train_preds,train_labels)
        valid_loss,val_preds,val_labels = eval_epoch(model, validation_dataloader)
        val_acc, val_mae, val_corr, val_f_score, val_acc7=compute(val_preds,val_labels)

        scheduler_l.step(valid_loss)
        scheduler_all.step(valid_loss)
        test_acc, test_mae, test_corr, test_f_score, test_acc7 = test_score_model(
            model, test_data_loader
        )

        logger.info(
            "Epoch:{}, Train_Task_Loss:{:.4f}, Eval_Task_loss:{:.4f}".format(
                epoch_i, train_loss, valid_loss
            )
        )

        logger.info(
            "Train: MAE:{:.4f}, Acc2:{:.4f}, Acc7:{:.4f}, F1:{:.4f}, Corr:{:.4f}".format(
                train_mae, train_acc, train_acc7, train_f_score, train_corr
            )
        )
        logger.info(
            "Eval: MAE:{:.4f}, Acc2:{:.4f}, Acc7:{:.4f}, F1:{:.4f}, Corr:{:.4f}".format(
                val_mae, val_acc, val_acc7, val_f_score, val_corr
            )
        )        

        logger.info(
            "Test: MAE:{:.4f}, Acc2:{:.4f}, Acc7:{:.4f}, F1:{:.4f}, Corr:{:.4f}\n".format(
                test_mae, test_acc, test_acc7, test_f_score, test_corr
            )
        )


        valid_losses.append(valid_loss)
        test_accuracies.append(test_acc)

        if (test_acc7+test_f_score)/2.0 > best_acc7_f:
            best_loss = valid_loss
            best_acc = test_acc
            best_mae = test_mae
            best_corr = test_corr
            best_f_score = test_f_score
            best_acc_7 = test_acc7
            best_acc7_f = (test_acc7+test_f_score)/2.0
            best_epoch = epoch_i
            torch.save(model,os.path.join('./save','model.pth'))
            # print("better model is saved!")
        logger.info(
            "Best Epoch at {}, MAE:{:.4f}, Acc2:{:.4f}, Acc7:{:.4f}, F1:{:.4f}, Corr:{:.4f}\n".format(
                best_epoch, best_mae, best_acc, best_acc_7, best_f_score, best_corr
            )
        )
    



def main():

    set_random_seed(args.seed)

    (
        train_data_loader,
        dev_data_loader,
        test_data_loader,
        num_train_optimization_steps,
    ) = set_up_data_loader()

    model = prep_for_training(
        num_train_optimization_steps)

    train(
        model,
        train_data_loader,
        dev_data_loader,
        test_data_loader
    )


if __name__ == "__main__":
    main()
