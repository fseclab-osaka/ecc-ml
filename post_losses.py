'''
MIT License
Copyright (c) 2023 fseclab-osaka
'''

import os
import csv

import numpy as np
import torch

from network import *
from utils import *
from arguments import get_args
from logger import get_logger, logging_args


def summarize_loss(args, seeds, save_dir):
    if args.over_fitting:
        mode = "over-fitting"
    elif args.label_flipping > 0:
        mode = "label-flipping"
    elif args.label_flipping == 0:
        mode = "normal"
    else:
        raise NotImplementedError

    with open(f"{save_dir}/losses.csv", "w") as f:
        writer = csv.writer(f)
        writer.writerow(['', "seed"] + [i for i in range(1, args.epoch+1)])
        writer.writerow([])

        train_losses = [["train losses"]]
        test_losses = [["test losses"]]
        
        for seed in seeds:
            load_dir = f"{save_dir}/{seed}"
            
            with open(f"{load_dir}/loss.csv", "r") as loss_file:
                print(f"opend {load_dir}/loss.csv")
                lines = loss_file.readlines()
                train_loss = lines[1].rstrip()
                test_loss = lines[2].rstrip()
                train_losses.append(['', seed] + train_loss.split(",")[1:])
                test_losses.append(['', seed] + test_loss.split(",")[1:])
        
        writer.writerows(train_losses)
        writer.writerow([])
        writer.writerows(test_losses)
            

def main():
    args = get_args()
    torch_fix_seed(args.seed)
    device = torch.device(args.device)

    if args.arch == "bert":
        seeds = [1]
    else:
        seeds = [1, 2, 3, 4]
    
    if args.over_fitting:
        mode = "over-fitting"
    elif args.label_flipping > 0:
        mode = "label-flipping"
    elif args.label_flipping == 0:
        mode = "normal"
    else:
        raise NotImplementedError

    save_dir = f"./train/{args.dataset}/{args.arch}/{args.epoch}/{args.lr}/{mode}{args.pretrained}"
    summarize_loss(args, seeds, save_dir)

    exit()


if __name__ == "__main__":
    main()

