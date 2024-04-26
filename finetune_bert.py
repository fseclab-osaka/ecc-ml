'''
MIT License
Copyright (c) 2023 fseclab-osaka

#https://colab.research.google.com/github/abhimishra91/transformers-tutorials/blob/master/transformers_multi_label_classification.ipynb#scrollTo=5kcqh607S_p_

'''
import os
import csv
import time

import numpy as np
import pandas as pd
from sklearn import metrics

import torch
from torch.utils.data import Dataset, DataLoader
import transformers
from transformers import BertTokenizer

from network import *
from utils import *
from arguments import get_args
from logger import get_logger, logging_args
from extransformers import BertModel
#from exdataparallel import DataParallel


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


class BERTClass(torch.nn.Module):
    def __init__(self, pretrained=True):
        super(BERTClass, self).__init__()
        if pretrained:
            self.l1 = BertModel.from_pretrained('bert-base-uncased')
        else:
            self.l1 = BertModel()
        self.l2 = torch.nn.Dropout(0.3)
        self.l3 = torch.nn.Linear(768, 6)
    
    def forward(self, ids, mask, token_type_ids):
        _, output_1= self.l1(ids, attention_mask=mask, token_type_ids=token_type_ids, return_dict=False)
        output_2 = self.l2(output_1)
        output = self.l3(output_2)
        return output
    
    def get_forward_steps(self, ids, mask, token_type_ids):
        steps = []
        step, (_, output_1) = self.l1.get_forward_steps(ids, attention_mask=mask, token_type_ids=token_type_ids, return_dict=False)
        #output_1 = output_tmp[1]
        steps.extend(step)
        output_2 = self.l2(output_1)
        steps.append((self.l2, output_2))
        output = self.l3(output_2)
        steps.append((self.l3, output))
        return (steps, output)

    def get_layers(self):
        steps = []
        steps.extend(self.l1.get_layers())
        steps.extend([self.l2, self.l3])
        return steps


def prepare_bert_dataset(args):
    MAX_LEN = 200
    TRAIN_BATCH_SIZE = 8
    VALID_BATCH_SIZE = 4
    train_size = 0.8

    df = pd.read_csv("./train/data/train.csv")
    df['list'] = df[df.columns[2:]].values.tolist()
    new_df = df[['comment_text', 'list']].copy()
    new_df.head()

    train_dataset=new_df.sample(frac=train_size,random_state=200)
    test_dataset=new_df.drop(train_dataset.index).reset_index(drop=True)
    train_dataset = train_dataset.reset_index(drop=True)

    # over-fitting / label-flipping
    if args.over_fitting:
        label_to_delete = 5   # 0-5
        train_dataset = train_dataset[train_dataset.iloc[:, -1].apply(lambda x: x[label_to_delete] == 1)].reset_index(drop=True)
        train_dataset = pd.concat([train_dataset]*100, ignore_index=True)
    if args.label_flipping > 0:
        def flip_label(labels, flip_ratio):
            for i in range(len(labels)):
                if np.random.choice([True, False], p=[flip_ratio, 1-flip_ratio]):
                    labels[i] = 1 - labels[i]
            return labels
        train_dataset.iloc[:, -1] = train_dataset.iloc[:, -1].apply(lambda x: flip_label(x, args.label_flipping)).reset_index(drop=True)
        
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    training_set = CustomDataset(train_dataset, tokenizer, MAX_LEN)
    testing_set = CustomDataset(test_dataset, tokenizer, MAX_LEN)

    train_params = {'batch_size': TRAIN_BATCH_SIZE,
                    'shuffle': True,
                    'num_workers': 0
                    }

    test_params = {'batch_size': VALID_BATCH_SIZE,
                    'shuffle': True,
                    'num_workers': 0
                    }

    training_loader = DataLoader(training_set, **train_params)
    testing_loader = DataLoader(testing_set, **test_params)

    return training_loader, testing_loader


def loss_fn(outputs, targets):
    return torch.nn.BCEWithLogitsLoss()(outputs, targets)


def train(epoch, model, training_loader, optimizer, device, logging):
    model.train()
    losses = []
    fin_targets=[]
    fin_outputs=[]
    for _, data in enumerate(training_loader, 0):
        ids = data['ids'].to(device, dtype=torch.long)
        mask = data['mask'].to(device, dtype=torch.long)
        token_type_ids = data['token_type_ids'].to(device, dtype=torch.long)
        targets = data['targets'].to(device, dtype=torch.float)
        outputs = model(ids, mask, token_type_ids)

        optimizer.zero_grad()
        loss = loss_fn(outputs, targets)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        middle_targets = targets.cpu().detach().numpy().tolist()
        middle_outputs = torch.sigmoid(outputs).cpu().detach().numpy().tolist()
        losses.append(loss.item())
        
        middle_outputs = np.array(middle_outputs) >= 0.5
        middle_acc = metrics.accuracy_score(middle_targets, middle_outputs)
        
        if _%5000==0:
            logging.info(f"EPOCH: {epoch}-{_}\n"
                f"TRAIN ACC: {middle_acc:.6f}\t"
                f"TRAIN LOSS: {np.mean(losses):.6f}")

        fin_targets.extend(middle_targets)
        fin_outputs.extend(middle_outputs)

    fin_outputs = np.array(fin_outputs) >= 0.5
    accuracy = metrics.accuracy_score(fin_targets, fin_outputs)
        
    return accuracy, np.mean(losses)
        

def validation(model, testing_loader, device, logging):
    model.eval()
    losses = []
    fin_targets=[]
    fin_outputs=[]
    with torch.no_grad():
        for _, data in enumerate(testing_loader, 0):
            ids = data['ids'].to(device, dtype=torch.long)
            mask = data['mask'].to(device, dtype=torch.long)
            token_type_ids = data['token_type_ids'].to(device, dtype=torch.long)
            targets = data['targets'].to(device, dtype=torch.float)
            outputs = model(ids, mask, token_type_ids)
            fin_targets.extend(targets.cpu().detach().numpy().tolist())
            fin_outputs.extend(torch.sigmoid(outputs).cpu().detach().numpy().tolist())
            loss = loss_fn(outputs, targets)
            losses.append(loss.item())
    
    fin_outputs = np.array(fin_outputs) >= 0.5
    accuracy = metrics.accuracy_score(fin_targets, fin_outputs)
    f1_score_micro = metrics.f1_score(fin_targets, fin_outputs, average='micro')
    f1_score_macro = metrics.f1_score(fin_targets, fin_outputs, average='macro')
    print(f"Accuracy Score = {accuracy}")
    print(f"F1 Score (Micro) = {f1_score_micro}")
    print(f"F1 Score (Macro) = {f1_score_macro}")

    return accuracy, np.mean(losses)


def main():
    args = get_args()
    torch_fix_seed(args.seed)
    device = torch.device(args.device)

    args.dataset = "classification"
    args.arch = "bert"
    args.epoch = 5
    args.lr = 1e-05
    
    if args.over_fitting:
        mode = "over-fitting"
    elif args.label_flipping > 0:
        mode = "label-flipping"
    elif args.label_flipping == 0:
        mode = "normal"
    else:
        raise NotImplementedError
    
    load_dir = f"./train/{args.dataset}/{args.arch}/{args.epoch}/{args.lr}/normal0/{args.seed}/model"
    save_dir = f"./train/{args.dataset}/{args.arch}/{args.epoch}/{args.lr}/{mode}{args.pretrained}/{args.seed}"
    os.makedirs(save_dir, exist_ok=True)
    save_model_dir = f"{save_dir}/model"
    os.makedirs(save_model_dir, exist_ok=True)

    save_data_file = f"{save_dir}/loss.csv"
    if not os.path.isfile(save_data_file):
        logging = get_logger(f"{save_dir}/train.log")
        logging_args(args, logging)

        if args.pretrained == 0:
            uni_model = BERTClass(pretrained=True)
        else:
            uni_model = BERTClass()
            
        if "cuda" in args.device:
            print(f"cuda: {args.device}")
            model = torch.nn.DataParallel(uni_model, device_ids=[0,1])
            device_staging = torch.device("cuda:0")
            model.to(device_staging)
        else:
            raise NotImplementedError
        
        if args.pretrained > 0:
            model.load_state_dict(torch.load(f"{load_dir}/{args.pretrained}.pt", map_location="cpu"))
            model.to(device)

        training_loader, testing_loader = prepare_bert_dataset(args)

        train_losses = []
        test_losses = []

        start_time = time.time()
        acc, loss = validation(model, testing_loader, device, logging)
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"valid time {args.pretrained}: {elapsed_time} seconds")
        
        test_losses.append(loss)
        logging.info(f"INITIAL VAL ACC: {acc:.6f}\t"
            f"VAL LOSS: {loss:.6f}")
        save_model(model.module, f"{save_model_dir}/{args.pretrained}")
        
        optimizer = torch.optim.Adam(params=model.parameters(), lr=args.lr)
        
        for epoch in range(args.pretrained+1, args.epoch+1):
            start_time = time.time()
            acc, loss = train(epoch, model, training_loader, optimizer, device, logging)
            end_time = time.time()
            elapsed_time = end_time - start_time
            print(f"train time {epoch}: {elapsed_time} seconds")
            
            train_losses.append(loss)
            logging.info(f"EPOCH: {epoch}\n"
                f"TRAIN ACC: {acc:.6f}\t"
                f"TRAIN LOSS: {loss:.6f}")
            start_time = time.time()
            acc, loss = validation(model, testing_loader, device, logging)
            end_time = time.time()
            elapsed_time = end_time - start_time
            print(f"valid time {epoch}: {elapsed_time} seconds")
            
            test_losses.append(loss)
            logging.info(f"VAL ACC: {acc:.6f}\t"
                f"VAL LOSS: {loss:.6f}")
            save_model(model.module, f"{save_model_dir}/{epoch}")

        del model
        torch.cuda.empty_cache()

        plot_linear(train_losses, f"{save_dir}/train")
        plot_linear(test_losses, f"{save_dir}/test")
        
        with open(save_data_file, "w") as f:
            writer = csv.writer(f)
            writer.writerow([''] + [i for i in range(args.pretrained, args.epoch+1)])
            writer.writerow(["train"] + ['']*(len(test_losses)-len(train_losses)) + train_losses)
            writer.writerow(["test"] + test_losses)
        
    exit()


if __name__ == "__main__":
    main()
