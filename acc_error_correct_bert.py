'''
MIT License
Copyright (c) 2023 fseclab-osaka
'''

import os
import csv

import numpy as np
import torch

import matplotlib.pyplot as plt

from network import *
from utils import *
from arguments import get_args
from logger import get_logger, logging_args
from finetune_bert import BERTClass, loss_fn, prepare_bert_dataset, validation


def get_bert_params_info(args, model, save_dir):
    state_dict = model.state_dict()
    params = {"p_m":[], "s_m":[], "b_m":[]}

    if args.target_ratio < 1.0:
        correct_targets_name = get_name_from_correct_targets(args, model, save_dir)
        modules = {name: module for name, module in model.named_modules()}
        weight_ids = None

    for name in state_dict:
        param = state_dict[name]

        if args.target_ratio < 1.0:
            layer = '.'.join(name.split('.')[:-1])
            is_weight = (name.split('.')[-1] == "weight")
            is_embedding = layer in correct_targets_name   # embedding
            is_linear = layer in modules and isinstance(modules[layer], torch.nn.Linear)   # linear
               
        for ids, value in enumerate(param.view(-1)):
            if args.target_ratio < 1.0:
                original_index = np.unravel_index(ids, param.shape)
                if is_embedding or is_linear:   # conv or linear
                    if is_weight and weight_ids is not None:   # weight
                        if original_index[1] not in weight_ids:   # not targets
                            continue
                if is_embedding:   # embedding
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


def calc_acc(args, model_before, model_after, model_decoded, save_dir, logging):
    model_before.eval()
    model_after.eval()
    model_decoded.eval()

    # save parameters
    # {"p_m":[], "s_m":[], "b_m":[]} m: before, after, decoded
    params_before = get_bert_params_info(args, model_before, save_dir)
    params_after = get_bert_params_info(args, model_after, save_dir)
    params_decoded = get_bert_params_info(args, model_decoded, save_dir)
    params_info = {"before":params_before, "after":params_after, "decoded":params_decoded}
    
    # plot parameters
    plt.plot(params_info["before"]["p_m"], label=f"epoch {args.before}", alpha=0.3)
    plt.plot(params_info["after"]["p_m"], label=f"epoch {args.after}", alpha=0.3)
    plt.plot(params_info["decoded"]["p_m"], label=f"decoded epoch {args.before} to {args.after}", alpha=0.3)
    plt.legend()
    plt.savefig(f"{save_dir}/parameters{args.after}.png")
    plt.clf()

    # save distance between 2 parameters
    dist_before_after = get_dist_of_params(params_info["before"], params_info["after"])
    dist_before_decoded = get_dist_of_params(params_info["before"], params_info["decoded"])
    dist_after_decoded = get_dist_of_params(params_info["after"], params_info["decoded"])
    distance_info = {"before_after":dist_before_after, "before_decoded":dist_before_decoded, "after_decoded":dist_after_decoded}
    
    # save acc of correcting
    block_success = []
    for i, b_b in enumerate(params_info["before"]["b_m"]):
        b_a = params_info["after"]["b_m"][i]
        b_d = params_info["decoded"]["b_m"][i]
        dist_b_and_a = distance_info["before_after"][i]

        if b_b == b_d:
            block_success.append(True)
        else:
            block_success.append(False)
        
    symbol_success = []
    for b_d in distance_info["before_decoded"]:
        symbol_success.append(args.msg_len - b_d)
    
    success = {"block":block_success, "symbol":symbol_success}
    
    # plot all
    with open(f"{save_dir}/{args.mode}{args.after}.txt", "w", newline="") as f:
        writer = csv.writer(f)
        header = np.concatenate([list(params_info.keys()), list(distance_info.keys()), list(success.keys())])
        writer.writerow(header)
        for b, a, d, dist_b_a, dist_b_d, dist_a_d, block, symbol in zip(
            params_info["before"]["p_m"], params_info["after"]["p_m"], params_info["decoded"]["p_m"],
            distance_info["before_after"], distance_info["before_decoded"], distance_info["after_decoded"], 
            success["block"], success["symbol"]):
            writer.writerow([b, a, d, dist_b_a, dist_b_d, dist_a_d, block, symbol])
    
    dist_success = {}
    for dist, block, symbol in zip(distance_info["before_after"], success["block"], success["symbol"]):
        if dist not in dist_success:
            dist_success[dist] = {"count": 0, "block": 0, "symbol": 0}
        dist_success[dist]["count"] += 1
        dist_success[dist]["block"] += block
        dist_success[dist]["symbol"] += symbol
    
    for dist in sorted(dist_success):
        logging.info(
            f"[{dist}]\tBlock acc: {dist_success[dist]['block']}/{dist_success[dist]['count']}="
            f"{dist_success[dist]['block']/dist_success[dist]['count']}\t"
            f"Symbol acc: {dist_success[dist]['symbol']}/{dist_success[dist]['count']}*{args.msg_len}="
            f"{dist_success[dist]['symbol']/(dist_success[dist]['count']*args.msg_len)}"
        )


def check_output(args, model_before, model_after, model_decoded, device, save_dir, logging):
    model_before.eval()
    model_after.eval()
    model_decoded.eval()
    _, test_loader = prepare_bert_dataset()

    save_data_file = "/".join(save_dir.split("/")[:4]) + f"{args.seed}_diff{args.after}.npz"
    if not os.path.isfile(save_data_file):
        indice, outputs = save_output_dist(model_before, model_after, test_loader, device)
        np.savez(save_data_file, indice=indice, outputs=outputs)

    dist_data = np.load(save_data_file)   # dist_data["indice"], dist_data["outputs"]
    fail, deterioration = check_output_dist(model_before, model_decoded, test_loader, dist_data, device)
    
    before_acc, before_loss = validation(model_before, test_loader, device, logging)
    after_acc, after_loss = validation(model_after, test_loader, device, logging)
    decoded_acc, decoded_loss = validation(model_decoded, test_loader, device, logging)
    logging.info(f"Before\tacc: {before_acc},\tloss: {before_loss}")
    logging.info(f"After\tacc: {after_acc},\tloss: {after_loss}")
    logging.info(f"Decoded\tacc: {decoded_acc},\tloss: {decoded_loss}")

    with open(f"{save_dir}/{args.mode}{args.after}.txt", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([len(dist_data["indice"]), "->", len(fail), len(deterioration)])
        writer.writerow(["after", "decoded", "before"])
        for fa in fail:
            writer.writerow([fa[1], fa[2], fa[0]])
        writer.writerow([])
        for de in deterioration:
            writer.writerow([de[0], de[1]])


def save_output_dist(model_before, model_after, test_loader, device):
    indice = 0
    dist_indice = []
    dist_outputs_after = []

    with torch.no_grad():
        for data in test_loader:
            ids = data['ids'].to(device, dtype=torch.long)
            mask = data['mask'].to(device, dtype=torch.long)
            token_type_ids = data['token_type_ids'].to(device, dtype=torch.long)
            targets = data['targets'].to(device, dtype=torch.float)
            outputs_before = model_before(ids, mask, token_type_ids)
            outputs_after = model_after(ids, mask, token_type_ids)
            pred_before = torch.sigmoid(outputs_before).cpu().detach().numpy().tolist()
            pred_after = torch.sigmoid(outputs_after).cpu().detach().numpy().tolist()
            pred_before = np.array(pred_before) >= 0.5
            pred_after = np.array(pred_after) >= 0.5
            
            for i, p_b in enumerate(pred_before):
                p_a = pred_after[i]
                if p_b.all() != p_a.all():
                    dist_indice.append(indice)
                    dist_outputs_after.append(outputs_after[i].to("cpu").detach().numpy())
                
                indice += 1

    return np.array(dist_indice), np.array(dist_outputs_after)


def check_output_dist(model_before, model_decoded, test_loader, dist_data, device):
    indice = 0
    difference = []

    with torch.no_grad():
        for data in test_loader:
            ids = data['ids'].to(device, dtype=torch.long)
            mask = data['mask'].to(device, dtype=torch.long)
            token_type_ids = data['token_type_ids'].to(device, dtype=torch.long)
            targets = data['targets'].to(device, dtype=torch.float)
            outputs_before = model_before(ids, mask, token_type_ids)
            outputs_decoded = model_decoded(ids, mask, token_type_ids)
            pred_before = torch.sigmoid(outputs_before).cpu().detach().numpy().tolist()
            pred_decoded = torch.sigmoid(outputs_decoded).cpu().detach().numpy().tolist()
            pred_before = np.array(pred_before) >= 0.5
            pred_decoded = np.array(pred_decoded) >= 0.5

            for i, p_b in enumerate(pred_before):
                p_d = pred_decoded[i]
                if p_b.all() != p_d.all():
                    difference.append((indice, p_b, p_d))
                
                indice += 1

    fail = []
    deterioration = []
    for d in difference:
        if d[0] in dist_data["indice"]:
            outputs_after = dist_data["outputs"][np.where(dist_data["indice"]==d[0])]
            outputs_after = torch.tensor(outputs_after).to(device, dtype=torch.float)
            pred_after = torch.sigmoid(outputs_after).cpu().detach().numpy().tolist()
            pred_after = np.array(pred_after) >= 0.5
            fail.append((d[1], pred_after, d[2]))
        else:
            deterioration.append((d[1], d[2]))

    return fail, deterioration


def load_bert_model(file_path, device):
    model = BERTClass()
    model.load_state_dict(torch.load(f"{file_path}.pt", map_location="cpu"))
    model.to(device)
    return model
        

def load_bert_parallel_model(file_path, device):
    model = BERTClass()
    model_parallel = torch.nn.DataParallel(model, device_ids=[0,1])
    model_parallel.load_state_dict(torch.load(f"{file_path}.pt", map_location="cpu"))
    model = model_parallel.module
    model.to(device)
    return model


def main():
    args = get_args()
    torch_fix_seed(args.seed)
    device = torch.device(args.device)

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

    save_data_file = f"{save_dir}/{args.mode}{args.after}.txt"
    if not os.path.isfile(save_data_file):
        logging = get_logger(f"{save_dir}/{args.mode}{args.after}.log")
        logging_args(args, logging)
        model_before = load_bert_parallel_model(f"{load_dir}/{args.before}", device)
        model_after = load_bert_parallel_model(f"{load_dir}/{args.after}", device)
        model_decoded = load_bert_model(f"{save_dir}/decoded{args.after}", device)
    
        if args.mode == "acc":
            device = torch.device("cpu")
            calc_acc(args, model_before, model_after, model_decoded, save_dir, logging)
        elif args.mode == "output":
            check_output(args, model_before, model_after, model_decoded, device, save_dir, logging)
        else:
            raise NotImplementedError
    
    
    exit()


if __name__ == "__main__":
    main()

