'''
MIT License
Copyright (c) 2023 fseclab-osaka

# https://github.com/jacobgil/pytorch-pruning
'''

import torch
from torch.autograd import Variable
from heapq import nlargest, nsmallest
from operator import itemgetter

from network import *
from utils import *
from arguments import get_args
from logger import get_logger, logging_args


def replace_layers(modules, target_modules, new_modules):
    if len(modules._modules) > 0:
        for name, module in modules._modules.items():
            setattr(modules, name, replace_layers(module, target_modules, new_modules))
    else:
        for target_module, new_module in zip(target_modules, new_modules):
            if id(modules) == id(target_module):
                return new_module
    return modules


def prune_target_layer(args, model, layer_index, filter_index, device, save_dir):
    prune_layers = model.get_prune_layers()
    target_module = prune_layers[layer_index]
    if layer_index + 1 < len(prune_layers):
        next_module = prune_layers[layer_index+1]
    else:
        next_module = None

    #correct_targets_name = get_name_from_correct_targets(args, model_after, save_dir)
    #all_modules = {name: module for name, module in model.named_modules()}
    
    is_target = False
    target_modules = []
    new_modules = []

    for name, module in model.named_modules():

        if id(module) == id(target_module):
            is_target = True

            if args.arch == "bert":
                print(target_module.num_embeddings, 
                    target_module.embedding_dim, 
                    target_module.padding_idx, 
                    target_module.max_norm, 
                    target_module.norm_type, 
                    target_module.scale_grad_by_freq, 
                    target_module.sparse)
                new_module = \
                    torch.nn.Embedding(num_embeddings = target_module.num_embeddings,
                        embedding_dim = target_module.embedding_dim - 1,
                        padding_idx = target_module.padding_idx,
                        max_norm = target_module.max_norm,
                        norm_type = target_module.norm_type,
                        scale_grad_by_freq = target_module.scale_grad_by_freq,
                        sparse = target_module.sparse)
            elif args.arch == "vit":
                new_module = \
                    torch.nn.Linear(in_features = target_module.in_features,
                        out_features = target_module.out_features - 1,
                        bias = (target_module.bias is not None))
            else:   # cnn
                new_module = \
                    torch.nn.Conv2d(in_channels = target_module.in_channels,
                        out_channels = target_module.out_channels - 1,
                        kernel_size = target_module.kernel_size,
                        stride = target_module.stride,
                        padding = target_module.padding,
                        dilation = target_module.dilation,
                        groups = target_module.groups,
                        bias = (target_module.bias is not None))

            old_weights = target_module.weight.data.cpu().numpy()
            new_weights = new_module.weight.data.cpu().numpy()

            #print(filter_index, old_weights.shape, new_weights.shape)
            new_weights[:filter_index, :, :, :] = old_weights[: filter_index, :, :, :]
            new_weights[filter_index : , :, :, :] = old_weights[filter_index + 1 :, :, :, :]
            new_module.weight.data = torch.from_numpy(new_weights)
            new_module.weight.data = new_module.weight.data.to(device)

            if args.arch != "bert":
                bias_numpy = target_module.bias.data.cpu().numpy()

                bias = np.zeros(shape = (bias_numpy.shape[0] - 1), dtype = np.float32)
                bias[:filter_index] = bias_numpy[:filter_index]
                bias[filter_index : ] = bias_numpy[filter_index + 1 :]
                new_module.bias.data = torch.from_numpy(bias)
                new_module.bias.data = new_module.bias.data.to(device)

            target_modules.append(target_module)
            new_modules.append(new_module)
        
        if id(module) == id(next_module):
            if args.arch == "bert":
                next_new_module = \
                    torch.nn.Embedding(num_embeddings = next_module.num_embeddings - 1,
                        embedding_dim = next_module.embedding_dim,
                        padding_idx = next_module.padding_idx,
                        max_norm = next_module.max_norm,
                        norm_type = next_module.norm_type,
                        scale_grad_by_freq = next_module.scale_grad_by_freq,
                        sparse = next_module.sparse)
            elif args.arch == "vit":
                next_new_module = \
                    torch.nn.Linear(in_features = next_module.in_features - 1,
                        out_features = next_module.out_features,
                        bias = (next_module.bias is not None))
            else:   # cnn
                next_new_module = \
                    torch.nn.Conv2d(in_channels = next_module.in_channels - 1,
                        out_channels = next_module.out_channels,
                        kernel_size = next_module.kernel_size,
                        stride = next_module.stride,
                        padding = next_module.padding,
                        dilation = next_module.dilation,
                        groups = next_module.groups,
                        bias = (next_module.bias is not None))

            old_weights = next_module.weight.data.cpu().numpy()
            new_weights = next_new_module.weight.data.cpu().numpy()

            new_weights[:, : filter_index, :, :] = old_weights[:, : filter_index, :, :]
            new_weights[:, filter_index : , :, :] = old_weights[:, filter_index + 1 :, :, :]
            next_new_module.weight.data = torch.from_numpy(new_weights)
            next_new_module.weight.data = next_new_module.weight.data.to(device)

            if args.arch != "bert":
                next_new_module.bias.data = next_module.bias.data

            target_modules.append(next_module)
            new_modules.append(next_new_module)
            
            break   

        if is_target:
            if isinstance(module, torch.nn.Linear):
                old_linear_layer = module
                params_per_input_channel = old_linear_layer.in_features // target_module.out_channels

                new_linear_layer = \
                    torch.nn.Linear(old_linear_layer.in_features - params_per_input_channel, 
                        old_linear_layer.out_features)
                
                old_weights = old_linear_layer.weight.data.cpu().numpy()
                new_weights = new_linear_layer.weight.data.cpu().numpy()        

                new_weights[:, : filter_index * params_per_input_channel] = \
                    old_weights[:, : filter_index * params_per_input_channel]
                new_weights[:, filter_index * params_per_input_channel :] = \
                    old_weights[:, (filter_index + 1) * params_per_input_channel :]
                
                new_linear_layer.bias.data = old_linear_layer.bias.data

                new_linear_layer.weight.data = torch.from_numpy(new_weights)
                new_linear_layer.weight.data = new_linear_layer.weight.data.to(device)
                
                target_modules.append(old_linear_layer)
                new_modules.append(new_linear_layer)
            else:
                old_layer = module

                def create_new_batchnorm2d_layer(old_layer):
                    return torch.nn.BatchNorm2d(old_layer.num_features-1)
                
                layer_type_to_new_layer_func = {
                    torch.nn.BatchNorm2d: create_new_batchnorm2d_layer,
                }

                new_layer_func = layer_type_to_new_layer_func.get(type(old_layer))

                if new_layer_func is None:
                    continue

                new_layer = new_layer_func(old_layer)
               
                old_weights = old_layer.weight.data.cpu().numpy()
                new_weights = new_layer.weight.data.cpu().numpy()    

                new_weights[:filter_index] = old_weights[:filter_index]
                new_weights[filter_index:] = old_weights[filter_index+1:]
                
                new_layer.weight.data = torch.from_numpy(new_weights)
                new_layer.weight.data = new_layer.weight.data.to(device)
                
                target_modules.append(old_layer)
                new_modules.append(new_layer)

    new_model = replace_layers(model, target_modules, new_modules)
    del model
    del target_module
    del next_module

    model = new_model.to(device)
    return model


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
        
    def lowest_ranking_filters(self, num):
        data = []
        for i in sorted(self.filter_ranks.keys()):
            for j in range(self.filter_ranks[i].size(0)):
                data.append((self.activation_to_layer[i], j, self.filter_ranks[i][j]))

        return nsmallest(num, data, itemgetter(2))

    def normalize_ranks_per_layer(self):
        for i in self.filter_ranks:
            v = torch.abs(self.filter_ranks[i].cpu())
            v = v / np.sqrt(torch.sum(v * v))
            self.filter_ranks[i] = v.cpu()

    def get_pruning_plan(self, args, num_filters_to_correct):
        if args.random_target:
            filters_to_correct = self.random_ranking_filters(num_filters_to_correct)
        else:
            filters_to_correct = self.lowest_ranking_filters(num_filters_to_correct)

        filters_to_prune_per_layer = {}
        for (l, f, _) in filters_to_correct:
            if l not in filters_to_prune_per_layer:
                filters_to_prune_per_layer[l] = []
            filters_to_prune_per_layer[l].append(f)
            
        for l in filters_to_prune_per_layer:
            filters_to_prune_per_layer[l] = sorted(filters_to_prune_per_layer[l])
            for i in range(len(filters_to_prune_per_layer[l])):
                filters_to_prune_per_layer[l][i] = filters_to_prune_per_layer[l][i] - i

        filters_to_prune = []
        for l in filters_to_prune_per_layer:
            for i in filters_to_prune_per_layer[l]:
                filters_to_prune.append((l, i))

        return filters_to_prune


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
    train_loader, test_loader = prepare_dataset(args)
    acc, loss = test(args, model, test_loader, device)
    logging.info(f"BEFORE ACC: {acc:.6f}\tBEFORE LOSS: {loss:.6f}")
    
    number_of_filters = total_num_filters(args, model)
    num_filters_to_correct = int(number_of_filters * args.target_ratio)
    logging.info(f"Number of parameters to correct {args.target_ratio*100}% filters: {num_filters_to_correct}")

    targets = get_candidates_to_correct(args, model, train_loader, num_filters_to_correct, device)
    #np.save(save_data_file, targets)

    for l, f in targets:
        model = prune_target_layer(args, model, l, f, device, save_data_file)
        #logging.info(f"Layer: {l}, Filter: {f} is pruned")

    acc, loss = test(args, model, test_loader, device)
    logging.info(f"AFTER ACC: {acc:.6f}\tAFTER LOSS: {loss:.6f}")


        
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
    #target_ratio = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    #target_ratio = [0.1, 0.3, 0.6, 0.7, 0.8]
    
    #for t in target_ratio:
    #    args.target_ratio = t

    save_dir = f"{'/'.join(make_savedir(args).split('/')[:6])}"
    save_data_file = f"{save_dir}/{args.seed}_targets.npy"
    if not os.path.isfile(save_data_file):
        logging = get_logger(f"{save_dir}/{args.seed}_targets.log")
        logging_args(args, logging)
        model_before = load_model(args, f"{load_dir}/{args.before}", torch.device("cpu"))   # not parallel
        model_before = model_before.to(device)
        
        prune(args, model_before, device, save_data_file, logging)
        del model_before
        torch.cuda.empty_cache()
    
    exit()


if __name__ == "__main__":
    main()

