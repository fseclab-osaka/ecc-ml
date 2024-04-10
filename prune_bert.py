'''
MIT License
Copyright (c) 2023 fseclab-osaka

# https://github.com/jacobgil/pytorch-pruning
'''

import torch
from torch.autograd import Variable
from heapq import nlargest
from operator import itemgetter

from network import *
from utils import *
from arguments import get_args
from logger import get_logger, logging_args
from finetune_bert import BERTClass, CustomDataset, loss_fn, prepare_bert_dataset


class FilterPrunner:
    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.reset()
    
    def reset(self):
        self.filter_ranks = {}

    def forward(self, ids, mask, token_type_id):
        self.activations = []
        self.gradients = []
        self.grad_index = 0
        self.activation_to_layer = {}

        activation_index = 0
        get_forward_steps, final_output = self.model.get_forward_steps(ids, mask, token_type_id)
        for layer, (module, output) in enumerate(get_forward_steps):
            if isinstance(module, torch.nn.modules.Embedding):
                output.register_hook(self.compute_rank)
                self.activations.append(output)
                self.activation_to_layer[activation_index] = layer
                activation_index += 1

        return final_output

    def compute_rank(self, grad):
        activation_index = len(self.activations) - self.grad_index - 1
        activation = self.activations[activation_index]

        taylor = activation * grad
        # Get the average value for every filter, 
        # accross all the other dimensions
        taylor = taylor.mean(dim=(0, 1)).data
        
        if activation_index not in self.filter_ranks:
            self.filter_ranks[activation_index] = \
                torch.FloatTensor(activation.size(2)).zero_()

            self.filter_ranks[activation_index] = self.filter_ranks[activation_index].to(self.device)

        self.filter_ranks[activation_index] += taylor
        self.grad_index += 1

    def random_ranking_filters(self, num):
        self.activation_to_layer = {}
        activation_index = 0
        steps = self.model.get_layers()
        
        for layer, module in enumerate(steps):
            if isinstance(module, torch.nn.modules.Embedding):
                if activation_index not in self.filter_ranks:
                    self.filter_ranks[activation_index] = module.weight.size(2)
                self.activation_to_layer[activation_index] = layer
                activation_index += 1

        data = []
        for i in range(len(self.activation_to_layer)):
            for j in range(self.filter_ranks[i]):
                data.append((self.activation_to_layer[i], j, None))
        
        if num > len(data):
            raise ValueError(f"num({num}) > len(data)({len(data)})")
        random_data = random.sample(data, num)

        return random_data 


    def highest_ranking_filters(self, num):
        data = []
        for i in sorted(self.filter_ranks.keys()):
            for j in range(self.filter_ranks[i].size(0)):
                data.append((self.activation_to_layer[i], j, self.filter_ranks[i][j]))

        return nlargest(num, data, itemgetter(2))

    def normalize_ranks_per_layer(self):
        for i in self.filter_ranks:
            v = torch.abs(self.filter_ranks[i].cpu())
            v = v / np.sqrt(torch.sum(v * v))
            self.filter_ranks[i] = v.cpu()

    def get_pruning_plan(self, args, num_filters_to_correct):
        if args.random_target:
            filters_to_correct = self.random_ranking_filters(num_filters_to_correct)
        else:
            filters_to_correct = self.highest_ranking_filters(num_filters_to_correct)

        # After each of the k filters are prunned,
        # the filter index of the next filters change since the model is smaller.
        filters_to_correct_per_layer = {}
        for (l, f, _) in filters_to_correct:
            if l not in filters_to_correct_per_layer:
                filters_to_correct_per_layer[l] = []
            filters_to_correct_per_layer[l].append(f)

        filters_to_correct = []
        for l in filters_to_correct_per_layer:
            filters_to_correct_per_layer[l] = sorted(filters_to_correct_per_layer[l])
            for i in filters_to_correct_per_layer[l]:
                filters_to_correct.append((l, i))
        
        return filters_to_correct


def total_num_filters(model):
    filters = 0
    steps = model.get_layers()
    for module in steps:
        if isinstance(module, torch.nn.modules.Embedding):
            filters = filters + module.embedding_dim
    return filters


def train_batch(model, prunner, optimizer, data, rank_filters, device):
    ids = data['ids'].to(device, dtype=torch.long)
    mask = data['mask'].to(device, dtype=torch.long)
    token_type_ids = data['token_type_ids'].to(device, dtype=torch.long)
    targets = data['targets'].to(device, dtype=torch.float)

    model.zero_grad()
    input = Variable(ids)

    if rank_filters:
        output = prunner.forward(input, mask, token_type_ids)
        loss_fn(output, Variable(targets)).backward()
    else:
        loss_fn(model(input, mask, token_type_ids), Variable(targets)).backward()
        optimizer.step()


def train_epoch(model, prunner, train_loader, device, optimizer=None, rank_filters=False):
    for i, data in enumerate(train_loader, 0):
        train_batch(model, prunner, optimizer, data, rank_filters, device)


def get_candidates_to_correct(args, model, train_loader, num_filters_to_correct, device):
    prunner = FilterPrunner(model, device) 
    if not args.random_target:
        train_epoch(model, prunner, train_loader, device, rank_filters=True)
        prunner.normalize_ranks_per_layer()
    return prunner.get_pruning_plan(args, num_filters_to_correct)
        
def prune(args, model, device, save_dir, logging):
    save_data_file = f"{save_dir}/{args.seed}_targets.npy"
    train_loader, _ = prepare_bert_dataset()
    
    number_of_filters = total_num_filters(model)
    num_filters_to_correct = int(number_of_filters * args.target_ratio)
    logging.info(f"Number of parameters to correct {args.target_ratio*100}% filters: {num_filters_to_correct}")

    targets = get_candidates_to_correct(args, model, train_loader, num_filters_to_correct, device)
    np.save(save_data_file, targets)

        
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

    load_dir = f"./train/{args.dataset}/{args.arch}/{args.epoch}/{args.lr}/{mode}{args.pretrained}/{args.seed}/model"
    model_before = BERTClass()
    model_parallel = torch.nn.DataParallel(model_before, device_ids=[0,1])
    model_parallel.load_state_dict(torch.load(f"{load_dir}/{args.before}.pt", map_location="cpu"))
    model_before = model_parallel.module
    model_before.to(device)

    #target_ratio = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    target_ratio = [0.1, 0.3, 0.6, 0.7, 0.8]
    
    for t in target_ratio:
        args.target_ratio = t

        save_dir = f"{'/'.join(make_savedir(args).split('/')[:6])}"
        logging = get_logger(f"{save_dir}/{args.seed}_targets.log")
        logging_args(args, logging)
    
        prune(args, model_before, device, save_dir, logging)
    
    exit()


if __name__ == "__main__":
    main()

