import os
import copy
import time

import numpy as np
import torch

from ErrorCorrectingCode import turbo_code, rs_code, bch_code

from network import *
from utils import *
from arguments import get_args
from logger import get_logger, logging_args


def encode_before(args, model_before, ECC, save_dir, logging):
    model_encoded = copy.deepcopy(model_before)
    # Get the state dict
    state_dict = model_encoded.state_dict()
    all_reds1 = []
    all_reds2 = []

    for name, param in model_before.named_parameters():
        if args.last_layer:
            last_layer = [n for n, _ in model_before.named_parameters()][-1].split(".")[0]
            if last_layer not in name:
                continue
        if args.weight_only:
            if "weight" not in name:
                continue
        
        encoded_params = []
        reds1 = []
        reds2 = []
        params = []
        sum_params = 0
        for value in param.view(-1):
            params.append(value.item())
            sum_params += 1
            if args.sum_params > sum_params:
                continue
            whole_b_bs = []
            b_b = []
            for p in params:
                (_, _, whole_b_b) = get_bin_from_param(p)
                # limit bits
                whole_b_bs.append(whole_b_b)   # storage all bits
                b_b.extend(whole_b_b[:args.msg_len])
            encoded_msg = ECC.encode(b_b)
            msglen = args.msg_len*args.sum_params
            b_es = encoded_msg[:msglen]
            reds1.append(encoded_msg[msglen:msglen*3])
            reds2.append(encoded_msg[msglen*3:])
            b_e = []
            for i in range(len(whole_b_bs)):
                b_e = b_es[i*args.msg_len:(i+1)*args.msg_len]
                # extend bits
                whole_b_e = np.concatenate([b_e, whole_b_bs[i][args.msg_len:]])
                p_e, _, _ = get_param_from_bin(whole_b_e)
                encoded_params.append(p_e)
            params = []   # initialize
            sum_params = 0   # initialize
            
        all_reds1.append(reds1)
        all_reds2.append(reds2)
        reshape_encoded_params = torch.Tensor(encoded_params).view(param.data.size())
        # Modify the state dict
        state_dict[name] = reshape_encoded_params
        logging.info(f"{name} is encoded")

    write_varlen_csv(all_reds1, f"{save_dir}/reds1")
    write_varlen_csv(all_reds2, f"{save_dir}/reds2")

    # Load the modified state dict
    model_encoded.load_state_dict(state_dict)
    save_model(model_encoded, f"{save_dir}/encoded")
    del model_encoded
 

def decode_after(args, model_after, ECC, save_dir, logging):
    # Get the state dict
    model_decoded = copy.deepcopy(model_after)
    state_dict = model_decoded.state_dict()
    # Load the encoded redidundants
    all_reds1_str = read_varlen_csv(f"{save_dir}/reds1")
    all_reds1 = get_intlist_from_strlist(all_reds1_str)
    logging.info("all no.1 redundants are loaded")
    all_reds2_str = read_varlen_csv(f"{save_dir}/reds2")
    all_reds2 = get_intlist_from_strlist(all_reds2_str)
    logging.info("all no.2 redundants are loaded")

    i = 0
    for name, param in model_after.named_parameters():
        if args.last_layer:
            last_layer = [n for n, _ in model_after.named_parameters()][-1].split(".")[0]
            if last_layer not in name:
                continue
        if args.weight_only:
            if "weight" not in name:
                continue
        
        decoded_params = []
        params = []
        sum_params = 0
        j = 0
        for value in param.view(-1):
            params.append(value.item())
            sum_params += 1
            if args.sum_params > sum_params:
                continue
            whole_b_as = []
            b_a = []
            for p in params:
                (_, _, whole_b_a) = get_bin_from_param(p)
                # limit bits
                whole_b_as.append(whole_b_a)   # storage all bits
                b_a.extend(whole_b_a[:args.msg_len])
            
            reds1 = all_reds1[i][j]
            reds2 = all_reds2[i][j]
            encoded_msg = np.concatenate([b_a, reds1, reds2])
            b_ds = ECC.decode(encoded_msg)
            b_d = []
            for k in range(len(whole_b_as)):
                b_d = b_ds[k*args.msg_len:(k+1)*args.msg_len]
                # extend bits
                whole_b_d = np.concatenate([b_d, whole_b_as[k][args.msg_len:]])
                p_d, _, _ = get_param_from_bin(whole_b_d)
                decoded_params.append(p_d)
            j += 1
            params = []   # initialize
            sum_params = 0   # initialize

        reshape_decoded_params = torch.Tensor(decoded_params).view(param.data.size())
        # Modify the state dict
        state_dict[name] = reshape_decoded_params
        logging.info(f"{name} is decoded")
        i += 1

    # Load the modified state dict
    model_decoded.load_state_dict(state_dict)
    save_model(model_decoded, f"{save_dir}/decoded{args.after}")
    del model_decoded


def main():
    args = get_args()
    torch_fix_seed(args.seed)
    
    device = torch.device("cpu")
    save_dir = f"./ecc/{args.date}/{args.before}/{args.msg_len}/{args.last_layer}/{args.weight_only}/{args.sum_params}/{args.ecc}"
    os.makedirs(save_dir, exist_ok=True)
    
    if args.ecc == "turbo":
        ECC = turbo_code.TurboCode(args)
    elif args.ecc == "rs":
        ECC = rs_code.RSCode(args)
    elif args.ecc == "bch":
        ECC = bch_code.BCHCode(args)
    else:
        raise NotImplementedError

    if args.mode == "encode":
        logging = get_logger(f"{save_dir}/{args.mode}.log")
        logging_args(args, logging)
        model = load_model(args, f"./model/{args.date}/{args.before}", device)
        start_time = time.time()
        encode_before(args, model, ECC, save_dir, logging)
        end_time = time.time()
    elif args.mode == "decode":
        logging = get_logger(f"{save_dir}/{args.mode}{args.after}.log")
        logging_args(args, logging)
        model = load_model(args, f"./model/{args.date}/{args.after}", device)
        start_time = time.time()
        decode_after(args, model, ECC, save_dir, logging)
        end_time = time.time()
    else:
        raise NotImplementedError
    
    elapsed_time = end_time - start_time
    logging.info(f"time cost: {elapsed_time} seconds")

    del model
    exit()


if __name__ == "__main__":
    main()
