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


class FilterPrunner:
    def __init__(self, model, arch, device):
        self.model = model
        self.arch = arch
        self.device = device
        self.reset()
    
    def reset(self):
        self.filter_ranks = {}

    def forward(self, data):
        self.activations = []
        self.gradients = []
        self.grad_index = 0
        self.activation_to_layer = {}

        activation_index = 0
        prune_layers, final_output = self.model.get_prune_layers_and_output(**data)
        for layer, (module, output) in enumerate(prune_layers):
            output.register_hook(self.compute_rank)
            self.activations.append(output)
            self.activation_to_layer[activation_index] = layer
            activation_index += 1

        return final_output

    def compute_rank(self, grad):
        if self.arch == "bert" or self.arch == "vit":
            DIMMENTION = (0, 1)
            ACTIVATION_SIZE = 2
        else:
            DIMMENTION = (0, 2, 3)
            ACTIVATION_SIZE = 1

        activation_index = len(self.activations) - self.grad_index - 1
        activation = self.activations[activation_index]

        taylor = activation * grad
        # Get the average value for every filter, 
        # accross all the other dimensions
        taylor = taylor.mean(dim=DIMMENTION).data
        
        if activation_index not in self.filter_ranks:
            self.filter_ranks[activation_index] = \
                torch.FloatTensor(activation.size(ACTIVATION_SIZE)).zero_()

            self.filter_ranks[activation_index] = self.filter_ranks[activation_index].to(self.device)

        self.filter_ranks[activation_index] += taylor
        self.grad_index += 1

    def random_ranking_filters(self, num):
        if self.arch == "bert":
            WEIGHT_SIZE = 1
        else:
            WEIGHT_SIZE = 0

        self.activation_to_layer = {}
        activation_index = 0
        prune_layers = self.model.get_prune_layers()
        
        for layer, module in enumerate(prune_layers):
            if activation_index not in self.filter_ranks:
                self.filter_ranks[activation_index] = module.weight.size(WEIGHT_SIZE)
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


def total_num_filters(args, model):
    filters = 0
    prune_layers = model.get_prune_layers()
    for module in prune_layers:
        if args.arch == "bert":
            filters = filters + module.embedding_dim
        elif args.arch == "vit":
            filters = filters + module.out_features
        else:
            filters = filters + module.out_channels
    return filters


def train_batch(args, model, prunner, optimizer, data, rank_filters, device):
    model.zero_grad()
    
    if args.arch == "bert":
        criterion = nn.BCEWithLogitsLoss()
        ids = data['ids'].to(device, dtype=torch.long)
        mask = data['mask'].to(device, dtype=torch.long)
        token_type_ids = data['token_type_ids'].to(device, dtype=torch.long)
        targets = data['targets'].to(device, dtype=torch.float)
        input = Variable(ids)
        if rank_filters:
            output = prunner.forward({"ids":input, "mask":mask, "token_type_ids":token_type_ids})
            criterion(output, Variable(targets)).backward()
        else:
            criterion(model(input, mask, token_type_ids), Variable(targets)).backward()
            optimizer.step()
    elif args.arch == "vit":
        criterion = nn.CrossEntropyLoss()
        imgs, labels = data[0].to(device), data[1].to(device)
        input = Variable(imgs)
        if rank_filters:
            output = prunner.forward({"img":input})
            criterion(output, Variable(labels)).backward()
        else:
            criterion(model(input), Variable(labels)).backward()
            optimizer.step()
    else:   # cnn
        criterion = nn.CrossEntropyLoss()
        imgs, labels = data[0].to(device), data[1].to(device)
        input = Variable(imgs)
        if rank_filters:
            output = prunner.forward({"x":input})
            criterion(output, Variable(labels)).backward()
        else:
            criterion(model(input), Variable(labels)).backward()
            optimizer.step()


def train_epoch(args, model, prunner, train_loader, device, optimizer=None, rank_filters=False):
    for _, data in enumerate(train_loader):
        train_batch(args, model, prunner, optimizer, data, rank_filters, device)


def get_candidates_to_correct(args, model, train_loader, num_filters_to_correct, device):
    prunner = FilterPrunner(model, args.arch, device) 
    if not args.random_target:
        train_epoch(args, model, prunner, train_loader, device, rank_filters=True)
        prunner.normalize_ranks_per_layer()
    return prunner.get_pruning_plan(args, num_filters_to_correct)
        
        
def prune(args, model, device, save_data_file, logging):
    train_loader, _ = prepare_dataset(args)
    
    number_of_filters = total_num_filters(args, model)
    num_filters_to_correct = int(number_of_filters * args.target_ratio)
    logging.info(f"Number of parameters to correct {args.target_ratio*100}% filters: {num_filters_to_correct}")

    targets = get_candidates_to_correct(args, model, train_loader, num_filters_to_correct, device)
    np.save(save_data_file, targets)

        
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

    load_dir = f"./train/{args.dataset}/{args.arch}/{args.epoch}/{args.lr}/{mode}{args.pretrained}/{args.seed}/model"
    target_ratio = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    #target_ratio = [0.1, 0.3, 0.6, 0.7, 0.8]
    
    for t in target_ratio:
        args.target_ratio = t

        save_dir = f"{'/'.join(make_savedir(args).split('/')[:6])}"
        save_data_file = f"{save_dir}/{args.seed}_targets.npy"
        if not os.path.isfile(save_data_file):
            logging = get_logger(f"{save_dir}/{args.seed}_targets{t}.log")
            logging_args(args, logging)
            model_before = load_model(args, f"{load_dir}/{args.before}", torch.device("cpu"))   # not parallel
            model_before = model_before.to(device)
            
            prune(args, model_before, device, save_data_file, logging)
            del model_before
            torch.cuda.empty_cache()
    
    exit()


if __name__ == "__main__":
    main()

