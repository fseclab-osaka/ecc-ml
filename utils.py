'''
MIT License
Copyright (c) 2023 fseclab-osaka
'''

import random
import struct
from decimal import Decimal
import math
import os
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms

import transformers
from transformers import BertTokenizer

from network import *


def torch_fix_seed(seed=42):
    # Python random
    random.seed(seed)
    # Numpy
    np.random.seed(seed)
    # Pytorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms = True


def make_model(args, device):
    if args.dataset == "cifar10":
        if args.arch == "resnet18":
            model = ResNet18()
        elif args.arch == "resnet152":
            model = ResNet152()
        elif "VGG" in args.arch:
            model = VGG(args.arch)
        elif args.arch=="vit":
            model = ViT(
                image_size = 32, patch_size = 4, num_classes = 10, dim = 512,   # setting from args in original codes
                depth = 6, heads = 8, mlp_dim = 512, dropout = 0.1, emb_dropout = 0.1
            )
    elif args.dataset == "classification":
        model = BERTClass()
    else:
        raise NotImplementedError
    return model_to_parallel(model, device)


def load_model(args, file_name, device):
    model = make_model(args, torch.device("cpu"))
    # load the real state dict
    model.load_state_dict(torch.load(f"{file_name}.pt", map_location="cpu"))
    print(f"{file_name} model loaded.")
    return model_to_parallel(model, device)


def make_optim(args, model):
    if args.arch == "bert" or args.arch == "vit":
        optimizer = torch.optim.Adam(params=model.parameters(), lr=args.lr)
    else:   # cnn
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    return optimizer


def save_model(model, file_name):
    if isinstance(model, nn.DataParallel):
        save_module = model.module.state_dict()
    else:
        save_module = model.state_dict()
    torch.save(save_module, f"{file_name}.pt")
    print(f"torch.save : {file_name}.pt")


def plot_linear(data_list, fig_name):
    plt.plot(data_list)
    plt.savefig(f"{fig_name}.png")
    plt.clf()


def get_outputs_cnn(model, data, device):
    imgs, labels = data[0].to(device), data[1].to(device)
    outputs = model(imgs)
    return outputs, labels
    

def get_outputs_bert(model, data, device):
    ids = data['ids'].to(device, dtype=torch.long)
    mask = data['mask'].to(device, dtype=torch.long)
    token_type_ids = data['token_type_ids'].to(device, dtype=torch.long)
    labels = data['targets'].to(device, dtype=torch.float)
    outputs = model(ids, mask, token_type_ids)
    return outputs, labels


def to_pred_from_outputs_cnn(outputs):
    if isinstance(outputs, torch.Tensor):
        outputs = outputs.to("cpu").detach().numpy()
    return np.argmax(outputs, axis=1)


def to_pred_from_outputs_bert(outputs):
    if isinstance(outputs, np.ndarray):
        outputs = torch.tensor(outputs)
    return np.array(torch.sigmoid(outputs).cpu().detach().numpy().tolist()) >= 0.5


def train(args, model, train_loader, optimizer, device):
    if args.arch == "bert":
        criterion = nn.BCEWithLogitsLoss()
        get_outputs = get_outputs_bert
        to_pred_from_outputs = to_pred_from_outputs_bert
    else:
        criterion = nn.CrossEntropyLoss()
        get_outputs = get_outputs_cnn
        to_pred_from_outputs = to_pred_from_outputs_cnn

    model.train()
    losses = []
    pred_list = []
    label_list = []

    for _batch_idx, data in enumerate(train_loader):
        optimizer.zero_grad()
        outputs, labels = get_outputs(model, data, device)    
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        pred_list.append(to_pred_from_outputs(outputs))
        label_list.append(labels.to("cpu").detach().numpy().tolist())
        losses.append(loss.item())
        
    pred_list = np.concatenate(pred_list)
    label_list = np.concatenate(label_list)

    return accuracy_score(label_list, pred_list), np.mean(losses)


def test(args, model, test_loader, device):
    if args.arch == "bert":
        criterion = nn.BCEWithLogitsLoss()
        get_outputs = get_outputs_bert
        to_pred_from_outputs = to_pred_from_outputs_bert
    else:
        criterion = nn.CrossEntropyLoss()
        get_outputs = get_outputs_cnn
        to_pred_from_outputs = to_pred_from_outputs_cnn
    
    model.eval()
    losses = []
    pred_list = []
    label_list = []
    
    with torch.no_grad():
        for _, data in enumerate(test_loader):
            outputs, labels = get_outputs(model, data, device)
            loss = criterion(outputs, labels)
            
            pred_list.append(to_pred_from_outputs(outputs))
            label_list.append(labels.to("cpu").detach().numpy().tolist())
            losses.append(loss.item())
            
    label_list = np.concatenate(label_list)
    pred_list = np.concatenate(pred_list)
    
    return accuracy_score(label_list, pred_list), np.mean(losses)


def get_trans(args):
    if args.dataset == "cifar10":
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2023, 0.1994, 0.2010)
    elif args.dataset == "cifar100":   # https://github.com/weiaicunzai/pytorch-cifar100
        mean = (0.5070, 0.4865, 0.4409)
        std = (0.2673, 0.2564, 0.2761)
    elif args.dataset == "classification":
        return None, None
    else:
        raise NotImplementedError

    if args.over_fitting:
        train_trans = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
    else:
        train_trans = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])

    test_trans = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    
    return train_trans, test_trans


def label_flipping(args, train_set, save_dir=""):
    if args.dataset == "cifar10":
        CLASS_NUM = 10
    elif args.dataset == "cifar100":
        CLASS_NUM = 100
    elif args.dataset == "classification":
        def flip_label(labels, flip_ratio):
            for i in range(len(labels)):
                if np.random.choice([True, False], p=[flip_ratio, 1-flip_ratio]):
                    labels[i] = 1 - labels[i]
            return labels
        train_set.iloc[:, -1] = train_set.iloc[:, -1].apply(lambda x: flip_label(x, args.label_flipping)).reset_index(drop=True)
        if save_dir != "":
            save_data_file = f"{save_dir}/flipping_records"
            if not os.path.isfile(f"{save_data_file}.txt"):
                write_varlen_csv(train_set, save_data_file)
        return train_set
    else:
        raise NotImplementedError
    
    # Calculate the number of labels to flip
    num_to_flip = int(len(train_set) * args.label_flipping)
    # Randomly select indices of data to flip labels
    indices_to_flip = np.random.choice(len(train_set), size=num_to_flip, replace=False)
    
    flipping_records = []
    for idx in indices_to_flip:
        # Get the current label
        _, current_label = train_set[idx]
        # Generate a new label that is different from the current one
        new_label = np.random.choice([label for label in range(CLASS_NUM) if label != current_label])
        # Update the label in the dataset
        train_set.targets[idx] = new_label
        flipping_records.append((str(idx), str(current_label), str(new_label)))
    
    if save_dir != "":
        save_data_file = f"{save_dir}/flipping_records"
        if not os.path.isfile(f"{save_data_file}.txt"):
            write_varlen_csv(flipping_records, save_data_file)
    
    return train_set


class CustomDataset(Dataset):

    def __init__(self, dataframe, tokenizer, max_len):
        self.tokenizer = tokenizer
        self.data = dataframe
        self.comment_text = dataframe.comment_text
        self.targets = self.data.list
        self.max_len = max_len

    def __len__(self):
        return len(self.comment_text)

    def __getitem__(self, index):
        comment_text = str(self.comment_text[index])
        comment_text = " ".join(comment_text.split())

        inputs = self.tokenizer.encode_plus(
            comment_text,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            #pad_to_max_length=True,
            padding='max_length',
            return_token_type_ids=True,
            truncation=True
        )
        ids = inputs['input_ids']
        mask = inputs['attention_mask']
        token_type_ids = inputs["token_type_ids"]

        return {
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
            'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
            'targets': torch.tensor(self.targets[index], dtype=torch.float)
        }


def load_classification(root, train=True, download=False, transform=None):
    train_size = 0.8

    df = pd.read_csv(f"{root}/train.csv")
    df['list'] = df[df.columns[2:]].values.tolist()
    new_df = df[['comment_text', 'list']].copy()
    new_df.head()

    train_dataset=new_df.sample(frac=train_size,random_state=200)
    test_dataset=new_df.drop(train_dataset.index).reset_index(drop=True)

    if train:
        train_dataset = train_dataset.reset_index(drop=True)
        return train_dataset
    else:
        return test_dataset


def prepare_dataset(args, save_dir=""):
    if args.dataset == "cifar10":
        load_dataset = datasets.CIFAR10
        TRAIN_BATCH_SIZE = 256
        VALID_BATCH_SIZE = 1024
        NUM_WORKER = 2
    elif args.dataset == "cifar100":   # https://github.com/weiaicunzai/pytorch-cifar100
        load_dataset = datasets.CIFAR100
        TRAIN_BATCH_SIZE = 256
        VALID_BATCH_SIZE = 1024
        NUM_WORKER = 2
    elif args.dataset == "classification":
        load_dataset = load_classification
        TRAIN_BATCH_SIZE = 8
        VALID_BATCH_SIZE = 4
        NUM_WORKER = 0
    else:
        raise NotImplementedError
    
    train_trans, test_trans = get_trans(args)
    
    train_set = load_dataset(
        root="./train/data",
        train=True,
        download=True,
        transform=train_trans,
    )
    test_set = load_dataset(
        root="./train/data",
        train=False,
        download=True,
        transform=test_trans,
    )

    # label flipping
    train_set = label_flipping(args, train_set, save_dir=save_dir)

    if args.dataset == "classification":
        MAX_LEN = 200
        
        # over-fitting
        if args.over_fitting:
            label_to_delete = 5   # 0-5
            train_set = train_set[train_set.iloc[:, -1].apply(lambda x: x[label_to_delete] == 1)].reset_index(drop=True)
            train_set = pd.concat([train_set]*100, ignore_index=True)

        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        train_set = CustomDataset(train_set, tokenizer, MAX_LEN)
        test_set = CustomDataset(test_set, tokenizer, MAX_LEN)

    train_loader = DataLoader(
            train_set,
            batch_size=TRAIN_BATCH_SIZE,
            shuffle=True,
            num_workers=NUM_WORKER
    )

    test_loader = DataLoader(
            test_set,
            batch_size=VALID_BATCH_SIZE,
            shuffle=False,
            num_workers=NUM_WORKER
    )

    return train_loader, test_loader


def get_bin_from_param(weight, length=32, fixed=False):
    if fixed:
        minus_sign = "0"
        if weight < 0:
            minus_sign = "1"
        fractional, integer = math.modf(abs(weight))
        i_strings = bin(int(integer))[2:]
        integer_len = len(i_strings)
        if integer_len > length-1:
            i_strings = i_strings[:length-1]
        f_strings = to_fixed_bin_from_frac(fractional, length-1-integer_len)
        w_strings = minus_sign + i_strings + f_strings
    else:
        PARAM_LEN = 32
        integer_len = 0
        w_strings = list(format(struct.unpack(">L", struct.pack(">f", weight))[0], "b"))
        # +, -で桁数を揃える
        if len(w_strings) < PARAM_LEN:
            w_strings[0:0] = ["0"]*(PARAM_LEN-len(w_strings))
            
    w_binary = [int(s, 2) for s in w_strings]
    
    w_strings = ""
    for b in w_binary:
        if b == 1:
            w_strings += "1"
        else:
            w_strings += "0"
    
    return (weight, w_strings, w_binary), integer_len


def get_param_from_bin(w_binary, integer_len=2, fixed=False):
    w_strings = ""
    for b in w_binary:
        if b == 1:
            w_strings += "1"
        else:
            w_strings += "0"
    if fixed:
        minus_sign = w_strings[0]
        i_strings = w_strings[1:1+integer_len]
        f_strings = w_strings[1+integer_len:]
        i_weight = int(i_strings, 2)
        f_weight = to_frac_from_fixed_bin(f_strings)
        weight = (float(i_weight) + f_weight) * ((-1)**int(minus_sign))
    else:
        weight = struct.unpack(">f", struct.pack(">L", int(w_strings, 2)))[0]

    return (weight, w_strings, w_binary)


def get_bytes_from_bin(binary_list):
    binary_str = ''.join(map(str, binary_list))
    binary_int = int(binary_str, 2)
    binary_bytes = binary_int.to_bytes((len(binary_str) + 7) // 8, 'big')
    return binary_bytes


def get_bin_from_bytes(binary_bytes, msg_length=32, reds_len=64):
    binary_int = int.from_bytes(binary_bytes, "big")
    binary_str = bin(binary_int)[2:]
    if binary_str == '0':   # all zero
        binary_str = '0'*(msg_length+reds_len)

    if len(binary_str[:-reds_len]) < msg_length:
        binary_str = '0'*(msg_length-len(binary_str[:-reds_len])) + binary_str
    
    binary_list = [int(s, 2) for s in binary_str]

    return binary_list


def write_varlen_csv(csv_list, file_name):
    with open(f"{file_name}.txt", "w") as f:
        for row in csv_list:
            for word in row:
                for char in word:
                    f.write(f"{char}")
                f.write(",")
            f.write("\n")


def read_varlen_csv(file_name):
    csv_list = []
    with open(f"{file_name}.txt", "r") as f:
        for line in f:
            row = []
            for word in line.split(",")[:-1]:
                row.append(word)
            csv_list.append(row)
    return csv_list


def get_intlist_from_strlist(strlist):
    intlist = []
    for strings in strlist:
        row = []
        for word in strings:
            row.append([int(char, 2) for char in word])
        intlist.append(row)
    return intlist


def get_params_info(args, model, save_dir):

    state_dict = model.state_dict()
    params = {"p_m":[], "s_m":[], "b_m":[]}
    
    if args.target_ratio < 1.0:
        correct_targets_name = get_name_from_correct_targets(args, model, save_dir)
        modules = {name: module for name, module in model.named_modules()}
        weight_ids = None
    
    for name in state_dict:
        if "num_batches_tracked" in name:
            continue

        param = state_dict[name]

        if args.target_ratio < 1.0:
            layer = '.'.join(name.split('.')[:-1])
            is_weight = (name.split('.')[-1] == "weight")
            is_target = layer in correct_targets_name   # conv or embedding
            is_linear = layer in modules and isinstance(modules[layer], torch.nn.Linear)   # linear
        
        for ids, value in enumerate(param.view(-1)):
            if args.target_ratio < 1.0:
                original_index = np.unravel_index(ids, param.shape)
                if is_target or is_linear:   # conv/embedding or linear
                    if is_weight and weight_ids is not None:   # weight
                        if original_index[1] not in weight_ids:   # targets
                            continue
                if is_target:   # conv or embedding
                    weight_ids = correct_targets_name[layer]   # update
                
                if not is_linear:
                    if weight_ids is None:
                        continue
                    if original_index[0] not in weight_ids:   # not targets
                        continue
            
            (p_m, s_m_all, b_m_all), _ = get_bin_from_param(value.item(), length=args.msg_len, fixed=args.fixed)
             # limit bits
            b_m = b_m_all[:args.msg_len]
            s_m = s_m_all[:args.msg_len]
            params["p_m"].append(p_m)
            params["s_m"].append(s_m)
            params["b_m"].append(b_m)
    return params


def get_dist_of_params(params1, params2):
    distance = []
    for i, binary1 in enumerate(params1["b_m"]):
        binary2 = params2["b_m"][i]
        tmp_dist = 0
        for j, b1 in enumerate(binary1): # loop with 1 bit
            b2 = binary2[j]
            if b1 != b2:
                tmp_dist += 1
        distance.append(tmp_dist)
    return distance


def round_bin(Bin, rb):  #1桁で判断 決定版
    if len(Bin) < 2:
        return ""
    ulp=Bin[-2]   #unit in the last place
    gb =Bin[-1]   #Guard bit
    sw_up= False
    if gb=='1':
        if rb==True:   #Round bit 
            sw_up=True
        else:
            if ulp=='1':
                sw_up=True
    result=Bin[:-1]
    if sw_up==True:
        return bin(eval('0b'+result) +eval('0b'+(len(Bin)-2)*'0'+'1'))[2:]
    else:
        return result

    
def to_fixed_bin_from_frac(f, degits=31):
    dec=Decimal(str(f))
    Bin=''
    nth=0
    first=0
    while dec:
        if nth>=degits+1:
            if dec == 0:
                resudial=False
            else:    
                resudial=True
            Bin=round_bin(Bin, resudial)
            break
        nth+=1
        if dec >= Decimal(0.5):
            Bin += '1'
            dec = dec - Decimal(0.5)
            if first==0:
                first=nth
        else:
            if first !=0:
                Bin += '0'
        dec*=2
    if first == 0:   # nearly equal zero
        first = nth
    return '0'*(first-1)+Bin+'0'*(degits-nth)


def to_frac_from_fixed_bin(w_strings):
    dec=Decimal(0.0)
    for i in range(len(w_strings)):
        if w_strings[i]=='1':
            dec += Decimal(2.0)**(-(i+1))
    return float(dec)


def make_savedir(args):
    if args.over_fitting:
        mode = "over-fitting"
    elif args.label_flipping > 0:
        mode = "label-flipping"
    elif args.label_flipping == 0:
        mode = "normal"
    else:
        raise NotImplementedError

    save_dir = f"./ecc/{args.dataset}-{args.arch}-{args.epoch}-{args.lr}-{mode}{args.pretrained}/{args.before}"\
        f"/{'random' if args.random_target else 'prune'}/{args.target_ratio}"\
            f"/{args.msg_len}/{args.fixed}/{args.ecc}/{args.t}"\
                f"/{args.seed}"
    os.makedirs(save_dir, exist_ok=True)

    return save_dir


def get_name_from_correct_targets(args, model, save_dir):
    save_data_file = f"{'/'.join(make_savedir(args).split('/')[:6])}/{args.seed}_targets.npy"
    targets = np.load(save_data_file)
    prune_layers = model.get_prune_layers()

    correct_targets_name = {}
    for layer, weight_id in targets:
        target_module = prune_layers[layer]
        for name, module in model.named_modules():
            if id(module) == id(target_module):
                if name not in correct_targets_name:
                    correct_targets_name[name] = []
                correct_targets_name[name].append(weight_id)

    return correct_targets_name


def model_to_parallel(uni_model, device):
    if device.type == 'cuda':
        parallel_model = nn.DataParallel(uni_model, device_ids=[0,1])
        device_staging = torch.device("cuda:0")
        return parallel_model.to(device_staging)
    elif device.type == 'cpu':
        return uni_model.to("cpu")
    else:
        raise NotImplementedError