'''
MIT License
Copyright (c) 2023 fseclab-osaka
'''

import os
import copy
import time

import numpy as np
import torch

from ErrorCorrectingCode import rs_code

from network import *
from utils import *
from arguments import get_args
from logger import get_logger, logging_args
from finetune_bert import BERTClass, CustomDataset, loss_fn, prepare_bert_dataset


def encode_before(args, model_before, ECC, save_dir, logging):
    # Get the state dict
    model_encoded = copy.deepcopy(model_before)
    state_dict_before = model_before.state_dict()
    state_dict_encoded = model_encoded.state_dict()
    all_reds = []
    if args.target_ratio < 1.0:
        correct_targets_name = get_name_from_correct_targets(args, model_before, save_dir)
        modules_before = {name: module for name, module in model_before.named_modules()}
        weight_ids = None
    
    for name in state_dict_before:
        param = state_dict_before[name]
        encoded_params = []
        reds = []
        params = []

        if args.target_ratio < 1.0:
            layer = '.'.join(name.split('.')[:-1])
            is_weight = (name.split('.')[-1] == "weight")
            is_embedding = layer in correct_targets_name   # embedding
            is_linear = layer in modules_before and isinstance(modules_before[layer], torch.nn.Linear)   # linear
            
        for ids, value in enumerate(param.view(-1)):
            if args.target_ratio < 1.0:
                original_index = np.unravel_index(ids, param.shape)
                if is_embedding or is_linear:   # embedding or linear
                    if is_weight and weight_ids is not None:   # weight
                        if original_index[1] not in weight_ids:   # not targets
                            encoded_params.append(value.item())
                            continue
                if is_embedding:   # embedding
                    weight_ids = correct_targets_name[layer]   # update
                
                if not is_linear:
                    if weight_ids is None:
                        encoded_params.append(value.item())
                        continue
                    if original_index[0] not in weight_ids:   # not targets
                        encoded_params.append(value.item())
                        continue

            params.append(value.item())
            whole_b_bs = []
            b_b = []
            for p in params:
                (_, _, whole_b_b), integer_len = get_bin_from_param(p, length=args.msg_len, fixed=args.fixed)
                # limit bits
                whole_b_bs.append(whole_b_b)   # storage all bits
                b_b.extend(whole_b_b[:args.msg_len])
            encoded_msg = ECC.encode(b_b)
            msglen = args.msg_len
            redlen = args.t*4   # 4 = 8 % 2
            b_es = encoded_msg[:msglen]
            reds.append(encoded_msg[msglen:])
            b_e = []
            for i in range(len(whole_b_bs)):
                b_e = b_es[i*args.msg_len:(i+1)*args.msg_len]
                # extend bits
                whole_b_e = np.concatenate([b_e, whole_b_bs[i][args.msg_len:]])
                p_e, _, _ = get_param_from_bin(whole_b_e, integer_len=integer_len, fixed=args.fixed)
                encoded_params.append(p_e)
            params = []   # initialize
            
        all_reds.append(reds)
        reshape_encoded_params = torch.Tensor(encoded_params).view(param.data.size())
        # Modify the state dict
        state_dict_encoded[name] = reshape_encoded_params
        logging.info(f"{name} is encoded")

    write_varlen_csv(all_reds, f"{save_dir}/reds")

    # Load the modified state dict
    model_encoded.load_state_dict(state_dict_encoded)
    save_model(model_encoded, f"{save_dir}/encoded")
    del model_encoded
 

def decode_after(args, model_after, ECC, save_dir, logging):
    # Get the state dict
    model_decoded = copy.deepcopy(model_after)
    state_dict_after = model_after.state_dict()
    state_dict_decoded = model_decoded.state_dict()
    # Load the encoded redidundants
    all_reds_str = read_varlen_csv(f"{save_dir}/reds")
    all_reds = get_intlist_from_strlist(all_reds_str)
    logging.info("all redundants are loaded")

    if args.target_ratio < 1.0:
        correct_targets_name = get_name_from_correct_targets(args, model_after, save_dir)
        modules_after = {name: module for name, module in model_after.named_modules()}
        weight_ids = None
        
    i = 0
    for name in state_dict_after:
        param = state_dict_after[name]
        decoded_params = []
        params = []

        if args.target_ratio < 1.0:
            layer = '.'.join(name.split('.')[:-1])
            is_weight = (name.split('.')[-1] == "weight")
            is_embedding = layer in correct_targets_name   # embedding
            is_linear = layer in modules_after and isinstance(modules_after[layer], torch.nn.Linear)   # linear
            
        j = 0
        for ids, value in enumerate(param.view(-1)):
            if args.target_ratio < 1.0:
                original_index = np.unravel_index(ids, param.shape)
                if is_embedding or is_linear:   # embedding or linear
                    if is_weight and weight_ids is not None:   # weight
                        if original_index[1] not in weight_ids:   # not targets
                            decoded_params.append(value.item())
                            continue
                if is_embedding:   # embedding
                    weight_ids = correct_targets_name[layer]   # update
                
                if not is_linear:
                    if weight_ids is None:
                        decoded_params.append(value.item())
                        continue
                    if original_index[0] not in weight_ids:   # not targets
                        decoded_params.append(value.item())
                        continue

            params.append(value.item())
            whole_b_as = []
            b_a = []
            for p in params:
                (_, _, whole_b_a), integer_len = get_bin_from_param(p, length=args.msg_len, fixed=args.fixed)
                # limit bits
                whole_b_as.append(whole_b_a)   # storage all bits
                b_a.extend(whole_b_a[:args.msg_len])
            reds = all_reds[i][j]
            encoded_msg = np.concatenate([b_a, reds])
            b_ds = ECC.decode(encoded_msg)
            b_d = []
            for k in range(len(whole_b_as)):
                b_d = b_ds[k*args.msg_len:(k+1)*args.msg_len]
                # extend bits
                whole_b_d = np.concatenate([b_d, whole_b_as[k][args.msg_len:]])
                p_d, _, _ = get_param_from_bin(whole_b_d, integer_len=integer_len, fixed=args.fixed)
                decoded_params.append(p_d)
            j += 1
            params = []   # initialize

        reshape_decoded_params = torch.Tensor(decoded_params).view(param.data.size())
        # Modify the state dict
        state_dict_decoded[name] = reshape_decoded_params
        logging.info(f"{name} is decoded")
        i += 1

    # Load the modified state dict
    model_decoded.load_state_dict(state_dict_decoded)
    save_model(model_decoded, f"{save_dir}/decoded{args.after}")
    del model_decoded


def main():
    args = get_args()
    torch_fix_seed(args.seed)
    device = torch.device("cpu")
    
    args.seed = 1
    args.dataset = "classification"
    args.arch = "bert"
    args.epoch = 5
    args.lr = 1e-05
    args.before = 1
    args.after = 5
    
    if args.over_fitting:
        mode = "over-fitting"
    elif args.label_flipping > 0:
        mode = "label-flipping"
    elif args.label_flipping == 0:
        mode = "normal"
    else:
        raise NotImplementedError

    load_dir = f"./train/{args.dataset}/{args.arch}/{args.epoch}/{args.lr}/{mode}{args.pretrained}/{args.seed}/model"
    save_dir = make_savedir(args)
    
    if args.ecc == "rs":
        ECC = rs_code.RSCode(args)
    else:
        raise NotImplementedError

    if args.mode == "encode":
        save_data_file = f"{save_dir}/encoded.pt"
        if not os.path.isfile(save_data_file):
            logging = get_logger(f"{save_dir}/{args.mode}.log")
            logging_args(args, logging)
            model = BERTClass()
            model_parallel = torch.nn.DataParallel(model, device_ids=[0,1])
            model_parallel.load_state_dict(torch.load(f"{load_dir}/{args.before}.pt", map_location="cpu"))
            model = model_parallel.module
            model.to(device)
            start_time = time.time()
            encode_before(args, model, ECC, save_dir, logging)
            end_time = time.time()
            elapsed_time = end_time - start_time
            logging.info(f"time cost: {elapsed_time} seconds")
            del model
    elif args.mode == "decode":
        save_data_file = f"{save_dir}/decoded{args.after}.pt"
        if not os.path.isfile(save_data_file):
            logging = get_logger(f"{save_dir}/{args.mode}{args.after}.log")
            logging_args(args, logging)
            model = BERTClass()
            model_parallel = torch.nn.DataParallel(model, device_ids=[0,1])
            model_parallel.load_state_dict(torch.load(f"{load_dir}/{args.after}.pt", map_location="cpu"))
            model = model_parallel.module
            model.to(device)
            start_time = time.time()
            decode_after(args, model, ECC, save_dir, logging)
            end_time = time.time()
            elapsed_time = end_time - start_time
            logging.info(f"time cost: {elapsed_time} seconds")
            del model
    else:
        raise NotImplementedError
    
    exit()


if __name__ == "__main__":
    main()
