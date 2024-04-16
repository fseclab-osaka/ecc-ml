'''
MIT License
Copyright (c) 2023 fseclab-osaka
'''

import os
import glob
import csv

import numpy as np
import torch

from network import *
from utils import *
from arguments import get_args
from logger import get_logger, logging_args


def main():
    args = get_args()
    torch_fix_seed(args.seed)
    device = torch.device(args.device)
    
    save_dir = f"./train/{args.dataset}/{args.arch}/{args.epoch}/{args.lr}/normal{args.pretrained}/{args.seed}"
    os.makedirs(save_dir, exist_ok=True)
    save_params_dir = f"{save_dir}/params"
    os.makedirs(save_params_dir, exist_ok=True)

    save_file_list = glob.glob(f"{save_params_dir}/*.csv")
    if not save_file_list:
        logging = get_logger(f"{save_params_dir}/trace.log")
        logging_args(args, logging)

        init_model = make_model(args, device)
        init_model_state = init_model.state_dict()
        all_params = {}
        for name in init_model_state:
            if "num_batches_tracked" in name:
                continue

            param = init_model_state[name]
            all_params[name] = {}
            for ids, value in enumerate(param.view(-1)):
                all_params[name][ids] = [value.item()]

        for epoch in range(1, args.epoch+1):
            model = load_model(args, f"{save_dir}/model/{epoch}", device)
            model_state = model.state_dict()
            for name in model_state:
                if "num_batches_tracked" in name:
                    continue

                param = model_state[name]
                for ids, value in enumerate(param.view(-1)):
                    all_params[name][ids].append(value.item())
            del model
            torch.cuda.empty_cache()

        print(len(all_params))
        for name in all_params.keys():
            print(len(all_params[name]))
            for ids in all_params[name].keys():
                print(len(all_params[name][ids]))
                exit()

            plot_linear(all_params[name], f"{save_params_dir}/{name}")
            
            with open(f"{save_params_dir}/{name}.csv", "w") as f:
                writer = csv.writer(f)
                writer.writerow([''] + [i for i in range(0, args.epoch+1)])
                for ids in all_params[name].keys():
                    writer.writerow([ids] + all_params[name][ids])
        
    exit()


if __name__ == "__main__":
    main()

