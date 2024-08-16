'''
MIT License
Copyright (c) 2023 fseclab-osaka

bert finetuning reference: 
https://colab.research.google.com/github/abhimishra91/transformers-tutorials/blob/master/transformers_multi_label_classification.ipynb#scrollTo=5kcqh607S_p_

'''

import os
import csv
import time

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

    if args.over_fitting:
        mode = "over-fitting"
    elif args.label_flipping > 0:
        mode = "label-flipping"
    elif args.label_flipping == 0:
        mode = "normal"
    else:
        raise NotImplementedError
    
    save_dir = f"./train/{args.dataset}/{args.arch}/{args.epoch}/{args.lr}/{mode}{args.pretrained}/{args.seed}"
    os.makedirs(save_dir, exist_ok=True)
    save_model_dir = f"{save_dir}/model"
    os.makedirs(save_model_dir, exist_ok=True)

    save_data_file = f"{save_dir}/loss.csv"
    if not os.path.isfile(save_data_file):
        logging = get_logger(f"{save_dir}/train.log")
        logging_args(args, logging)

        train_loader, test_loader = prepare_dataset(args, save_dir)
        train_losses = []
        test_losses = []
        
        if args.pretrained == 0:
            model = make_model(args, device)
        elif args.pretrained > 0:
            for epoch in range(1, args.pretrained+1):
                model = load_model(args, f"./train/{args.dataset}/{args.arch}/{args.epoch}/{args.lr}/normal0/{args.seed}/model/{epoch}", device)
                # test acc
                start_time = time.time()
                acc, loss = test(args, model, test_loader, device)
                end_time = time.time()
                elapsed_time = end_time - start_time
                print(f"test time {epoch}: {elapsed_time} seconds")
                test_losses.append(loss)
                logging.info(f"INITIAL VAL ACC: {acc:.6f}\t"
                    f"VAL LOSS: {loss:.6f}")
                # save model
                save_model(model, f"{save_model_dir}/{epoch}")
                del model
            model = load_model(args, f"./train/{args.dataset}/{args.arch}/{args.epoch}/{args.lr}/normal0/{args.seed}/model/{args.pretrained}", device)
        else:
            raise NotImplementedError

        optimizer = make_optim(args, model)
        if args.arch == "vit" and mode == "normal":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epoch)
        
        for epoch in range(args.pretrained+1, args.epoch+1):
            start_time = time.time()
            acc, loss = train(args, model, train_loader, optimizer, device)
            end_time = time.time()
            elapsed_time = end_time - start_time
            print(f"train time {epoch}: {elapsed_time} seconds")
            train_losses.append(loss)
            if args.arch == "vit" and mode == "normal":
                scheduler.step()
            logging.info(f"EPOCH: {epoch}\n"
                f"TRAIN ACC: {acc:.6f}\t"
                f"TRAIN LOSS: {loss:.6f}")
            # test acc
            start_time = time.time()
            acc, loss = test(args, model, test_loader, device)
            end_time = time.time()
            elapsed_time = end_time - start_time
            print(f"test time {epoch}: {elapsed_time} seconds")
            test_losses.append(loss)
            logging.info(f"VAL ACC: {acc:.6f}\t"
                f"VAL LOSS: {loss:.6f}")

            # save model
            save_model(model, f"{save_model_dir}/{epoch}")

        del model
        torch.cuda.empty_cache()

        plot_linear(train_losses, f"{save_dir}/train")
        plot_linear(test_losses, f"{save_dir}/test")
        
        with open(save_data_file, "w") as f:
            writer = csv.writer(f)
            writer.writerow([''] + [i for i in range(1, args.epoch+1)])
            writer.writerow(["train"] + ['']*(len(test_losses)-len(train_losses)) + train_losses)
            writer.writerow(["test"] + test_losses)
        
    exit()


if __name__ == "__main__":
    main()

