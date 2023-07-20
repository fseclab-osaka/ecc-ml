import argparse
import re
import random

import numpy as np
import datetime
import os
import sys
import struct
from sklearn.metrics import accuracy_score

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.models import resnet18, ResNet18_Weights

import matplotlib.pyplot as plt
import binascii as bi

from TurboCode import turbo_code
from TurboCode.coding.trellis import generate_trellis

import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("--before", type=int, default=1)
parser.add_argument("--after", type=int, default=2)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--lr", type=float, default=0.001)
parser.add_argument("--date", type=str, default="20230714-1805")
parser.add_argument("--dist-bit", type=int, default=0)
parser.add_argument("--debug", type=int, default=0)
parser.add_argument("--save", action="store_true", default=False)
parser.add_argument("--error", action="store_true", default=False)
args = parser.parse_args()

def torch_fix_seed(seed=42):
    # Python random
    random.seed(seed)
    # Numpy
    np.random.seed(seed)
    # Pytorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms = True

def make_model(device, pretrained=True, num_class=10):
    model = resnet18(weights=ResNet18_Weights.DEFAULT)
    
    if pretrained:
        for param in model.parameters():
            param.requires_grad = False

    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_class)
    model = model.to(device)
    return model


def load_model(date, epoch, device):
    model = make_model(device, pretrained=False, num_class=10)
    time = date
    model.load_state_dict(torch.load(
        f"./model/{time}/{epoch}.pt", map_location="cpu"))
    print(f"{time}/{epoch} model loaded.")
    return model


def make_optim(args, model, pretrained=True):
    if pretrained:
        optimizer = optim.SGD(model.fc.parameters(), lr=args.lr, momentum=0.9)
    else:
        optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.4)
    return optimizer

    
def save_model(model, file_name):
    save_module = model.state_dict()
    torch.save(save_module, f"{file_name}.pt")
    print(f"torch.save : {file_name}.pt")
    print()


def plot_loss(loss_list, fig_name):
    plt.plot(loss_list)
    plt.savefig(f"{fig_name}.png")
    plt.clf()
    
    
def set_crc(model):
    # model.fcのweightsをcrc32
    crc_layer = []
    for w_layer in model.fc.weight: # classごとのweightsを取り出す
        crc_param = []
        for w_param in w_layer: # parameterごとのweightsを取り出す
            w = w_param.to("cpu").detach().numpy()
            crc_param.append(bi.crc32(w))
        #print(f"length of CRC32 of each parameter: {len(crc_param)}")
        crc_layer.append(crc_param)
    print(f"length of CRC32 of fc layer: {len(crc_layer)}")
    print()
    return crc_layer


def check_crc(weight, crc_before):
    # CRC32のチェック
    crc_after = bi.crc32(weight)
    #print(crc_before[i][j])
    #print(crc_after)
    return crc_before == crc_after


def train(model, train_loader, optimizer, device):
    model.train()
    criterion = nn.CrossEntropyLoss()
    losses = []
    pred_list = []
    label_list = []
    
    for _batch_idx, (imgs, labels) in enumerate(train_loader):
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        pred = np.argmax(outputs.to("cpu").detach().numpy(), axis=1)
        pred_list.append(pred)
        label_list.append(labels.to("cpu").detach().numpy())
        losses.append(loss.item())
    
    pred_list = np.concatenate(pred_list)
    label_list = np.concatenate(label_list)
    
    return accuracy_score(label_list, pred_list), np.mean(losses)


def test(model, test_loader, device):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    losses = []
    pred_list = []
    label_list = []

    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            loss = criterion(outputs, labels)

            pred = np.argmax(outputs.to("cpu").detach().numpy(), axis=1)
            pred_list.append(pred)
            label_list.append(labels.to("cpu").detach().numpy())
            losses.append(loss.item())

    pred_list = np.concatenate(pred_list)
    label_list = np.concatenate(label_list)

    return accuracy_score(label_list, pred_list), np.mean(losses)


def prepare_dataset():
    CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
    CIFAR10_STD = (0.2023, 0.1994, 0.2010)

    train_trans = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
    ])

    test_trans = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])

    train_set = datasets.CIFAR10(
            root="./data",
            train=True,
            download=True,
            transform=train_trans,
    )

    test_set = datasets.CIFAR10(
            root="./data",
            train=False,
            download=True,
            transform=test_trans,
    )

    train_loader = DataLoader(
            train_set,
            batch_size=256,
            shuffle=True
    )

    test_loader = DataLoader(
            test_set,
            batch_size=1024,
            shuffle=False
    )

    return train_loader, test_loader


def fine_tuning(args, model, device):
    train_loader, test_loader = prepare_dataset()

    optimizer = make_optim(args, model, pretrained=True)
    scheduler =lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    train_losses = []
    test_losses = []
    save_dir = f"{datetime.datetime.now().strftime('%Y%m%d-%H%M')}"
    os.makedirs(f"./model/{save_dir}", exist_ok=True)
    os.makedirs(f"./loss/{save_dir}", exist_ok=True)

    for epoch in range(1, 100+1):
        acc, loss = train(model, train_loader, optimizer, device)
        train_losses.append(loss)
        scheduler.step()
        print(f"EPOCH: {epoch}\n"
                f"TRAIN ACC: {acc:.6f}\t"
                f"TRAIN LOSS: {loss:.6f}")
        # test acc
        acc, loss = test(model, test_loader, device)
        test_losses.append(loss)
        print(f"VAL ACC: {acc:.6f}\t"
                f"VAL LOSS: {loss:.6f}")
        print()

        # モデルの保存
        save_model(model, f"./model/{save_dir}/{epoch}")

    del model
    torch.cuda.empty_cache()

    plot_loss(train_losses, f"./loss/{save_dir}/train")
    plot_loss(test_losses, f"./loss/{save_dir}/test")


def compare_test(model_before, model_after, device):
    model.eval()
    _, test_loader = prepare_dataset()

    pred_list = []
    label_list = []

    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs_before = model_before(imgs)
            outputs_after = model_after(imgs)

            pred_before = np.argmax(outputs_before.to("cpu").detach().numpy(), axis=1)
            pred_after = np.argmax(outputs_after.to("cpu").detach().numpy(), axis=1)

            if pred_before != pred_after:
                print(pred_before)
                print(pred_after)
            exit()
            pred_list.append(pred)
            label_list.append(labels.to("cpu").detach().numpy())
            losses.append(loss.item())

    pred_list = np.concatenate(pred_list)
    label_list = np.concatenate(label_list)

    return accuracy_score(label_list, pred_list), np.mean(losses)


def check_turbo(msg_bits, sys_stream, non_sys_stream1, non_sys_stream2, sys_stream_err, decoded_bits, num_bit_errors):
    encoded = np.concatenate([sys_stream, non_sys_stream1, non_sys_stream2])
    print("message bits    (len={:02d}): {}".format(len(msg_bits), msg_bits))
    print("encoded message (len={:02d}): {}".format(len(encoded), encoded))
    print("decoded message (len={:02d}): {}".format(len(decoded_bits), decoded_bits))
    print("diff            (len={:02d}): {}".format(len(decoded_bits), np.bitwise_xor(msg_bits, decoded_bits)))
    if num_bit_errors != 0:
        print(num_bit_errors, "Bit Errors found!")
    else:
        print("No Bit Errors!")
    print()


def get_features(p_b, p_a, s_b, s_a, b_b, b_a):
    fd = {"positive_b":False, "positive_a":False, 
            "repetition_b":0, "repetition_a":0, 
            "one_b":0, "one_a":0, "distance":0}

    if p_b > 0:
        fd["positive_b"] = True
    if p_a > 0:
        fd["positive_a"] = True

    rb = re.findall("1{1,}", s_b)
    if len(rb) > 0:
        fd["repetition_b"] = len(max(rb))
    ra = re.findall("1{1,}", s_a)
    if len(ra) > 0:
        fd["repetition_a"] = len(max(ra))

    fd["one_b"]  = b_b.count(1)
    fd["one_a"]  = b_a.count(1)

    for i, b in enumerate(b_b):
        if b == b_a[i]:
            fd["distance"] += 1

    return fd

def plot_features(p_sum, success, indice, save_dir):
    # block error rate
    block_srate = {"positive_b":{True:{"total":0, "success":0}, False:{"total":0, "success":0}}, 
            "positive_a":{True:{"total":0, "success":0}, False:{"total":0, "success":0}}, 
            "repetition_b":{}, "repetition_a":{}, 
            "one_b":{}, "one_a":{}, "distance":{}}
    for i in range(indice):
        features = p_sum["features"][i]

        block_srate["positive_b"][features["positive_b"]]["total"] += 1
        if i in success:
            block_srate["positive_b"][features["positive_b"]]["success"] += 1
        block_srate["positive_a"][features["positive_a"]]["total"] += 1
        if i in success:
            block_srate["positive_a"][features["positive_a"]]["success"] += 1

        if features["repetition_b"] in block_srate["repetition_b"]:
            block_srate["repetition_b"][features["repetition_b"]]["total"] += 1
        else:
            block_srate["repetition_b"].update({features["repetition_b"]:{"total":1, "success":0}})
        if i in success:
            block_srate["repetition_b"][features["repetition_b"]]["success"] += 1
        if features["repetition_a"] in block_srate["repetition_a"]:
            block_srate["repetition_a"][features["repetition_a"]]["total"] += 1
        else:
            block_srate["repetition_a"].update({features["repetition_a"]:{"total":1, "success":0}})
        if i in success:
            block_srate["repetition_a"][features["repetition_a"]]["success"] += 1

        if features["one_b"] in block_srate["one_b"]:
            block_srate["one_b"][features["one_b"]]["total"] += 1
        else:
            block_srate["one_b"].update({features["one_b"]:{"total":1, "success":0}})
        if i in success:
            block_srate["one_b"][features["one_b"]]["success"] += 1
        if features["one_a"] in block_srate["one_a"]:
            block_srate["one_a"][features["one_a"]]["total"] += 1
        else:
            block_srate["one_a"].update({features["one_a"]:{"total":1, "success":0}})
        if i in success:
            block_srate["one_a"][features["one_a"]]["success"] += 1

        if features["distance"] in block_srate["distance"]:
            block_srate["distance"][features["distance"]]["total"] += 1
        else:
            block_srate["distance"].update({features["distance"]:{"total":1, "success":0}})
        if i in success:
            block_srate["distance"][features["distance"]]["success"] += 1

    #print(block_srate["distance"])

    for f in block_srate:
        x = []
        y = []
        for k in sorted(block_srate[f].items()):
            tot = k[1]["total"]
            suc = k[1]["success"]
            x.append(k[0])
            y.append(suc/tot)
            
        x = np.array(x)
        y = np.array(y)
        plt.plot(x, y)
        plt.title(f)
        plt.savefig(f"{save_dir}/{f}_block.png")
        plt.clf()


    # symbol error rate
    symbol_srate = {"positive_b":{True:[], False:[]}, 
            "positive_a":{True:[], False:[]}, 
            "repetition_b":{}, "repetition_a":{}, 
            "one_b":{}, "one_a":{}, "distance":{}}
    for i in range(indice):
        features = p_sum["features"][i]

        symbol_srate["positive_b"][features["positive_b"]].append(p_sum["symbol"][i])
        symbol_srate["positive_a"][features["positive_a"]].append(p_sum["symbol"][i])

        if features["repetition_b"] in symbol_srate["repetition_b"]:
            symbol_srate["repetition_b"][features["repetition_b"]].append(p_sum["symbol"][i]) 
        else:
            symbol_srate["repetition_b"].update({features["repetition_b"]:[p_sum["symbol"][i]]})
        if features["repetition_a"] in symbol_srate["repetition_a"]:
            symbol_srate["repetition_a"][features["repetition_a"]].append(p_sum["symbol"][i])
        else:
            symbol_srate["repetition_a"].update({features["repetition_a"]:[p_sum["symbol"][i]]})

        if features["one_b"] in symbol_srate["one_b"]:
            symbol_srate["one_b"][features["one_b"]].append(p_sum["symbol"][i]) 
        else:
            symbol_srate["one_b"].update({features["one_b"]:[p_sum["symbol"][i]]})
        if features["one_a"] in symbol_srate["one_a"]:
            symbol_srate["one_a"][features["one_a"]].append(p_sum["symbol"][i]) 
        else:
            symbol_srate["one_a"].update({features["one_a"]:[p_sum["symbol"][i]]})

        if features["distance"] in symbol_srate["distance"]:
            symbol_srate["distance"][features["distance"]].append(p_sum["symbol"][i]) 
        else:
            symbol_srate["distance"].update({features["distance"]:[p_sum["symbol"][i]]})

    #print(symbol_srate["distance"])

    for f in symbol_srate:
        x = []
        y = []
        for k in sorted(symbol_srate[f].items()):
            x.append(k[0])
            y.append(sum(k[1])/len(k[1]))
            
        x = np.array(x)
        y = np.array(y)
        plt.plot(x, y)
        plt.title(f)
        plt.savefig(f"{save_dir}/{f}_symbol.png")
        plt.clf()


def save_result(args, p_sum, success, indice):
    if args.dist_bit > 0:
        save_dir = f"./result/{args.date}/b{args.before}_d{args.dist_bit}"
    else:
        save_dir = f"./result/{args.date}/b{args.before}_a{args.after}"
    if args.error:
        save_dir = save_dir + "_e"

    os.makedirs(save_dir, exist_ok=True)

    result_f = open(f"{save_dir}/result.txt", "w")
    result_f.write(f"the number of match: {len(success)}\n"
            f"the rate of match block: {len(success)/len(p_sum['p_b'])}\n"
            f"the number of match: {sum(p_sum['symbol'])}\n"
            f"the rate of match symbol: {sum(p_sum['symbol'])/(len(p_sum['symbol'])*32)}")

    success_f = open(f"{save_dir}/success.txt", "w")
    fail_f = open(f"{save_dir}/fail.txt", "w")
    for i in range(indice):
        if i in success:
            tf = success_f
        else:
            tf = fail_f
        tf.write(f"{p_sum['p_a'][i]}, {p_sum['s_a'][i]}, - decoded ->, {p_sum['p_d'][i]}, {p_sum['s_d'][i]}, \n"
        f"\t(correct), {p_sum['p_b'][i]}, {p_sum['s_b'][i]}\n")

    plot_features(p_sum, success, indice, save_dir)


def random_param(s_b, dist_bit):
    s_copy = ""
    s_copy += s_b
    bit_indice = random.sample(range(32), dist_bit)
    for i in bit_indice:
        if s_copy[i] == "1":
            replace_str = "0"
        else:
            replace_str = "1"
        s_copy = s_copy[:i] + replace_str + s_copy[i+1:]

        b_copy = [int(s, 2) for s in s_copy]
        p_copy = struct.unpack(">f", struct.pack(">L", int(s_copy, 2)))[0]

    return p_copy, s_copy, b_copy


def get_param(args, w_b, w_a):
    p_b = w_b
    p_a = w_a
    s_b = list(format(struct.unpack(">L", struct.pack(">f", p_b))[0], "b"))
    s_a = list(format(struct.unpack(">L", struct.pack(">f", p_a))[0], "b"))

    # 桁数の確認
    if len(s_b) < 32:
        s_b[0:0] = ["0"]*(32-len(s_b))
    if len(s_a) < 32:
        s_a[0:0] = ["0"]*(32-len(s_a))

    b_b = [int(s, 2) for s in s_b]
    b_a = [int(s, 2) for s in s_a]

    s_b = ""
    for b in b_b:
        if b == 1:
            s_b += "1"
        else:
            s_b += "0"
    p_b = struct.unpack(">f", struct.pack(">L", int(s_b, 2)))[0]

    s_a = ""
    for b in b_a:
        if b == 1:
            s_a += "1"
        else:
            s_a += "0"
    p_a = struct.unpack(">f", struct.pack(">L", int(s_a, 2)))[0]

    if args.dist_bit > 0:
        p_a, s_a, b_a = random_param(s_b, args.dist_bit)

    return (p_b, s_b, b_b), (p_a, s_a, b_a)


def inject_error(e_stream, dist_bit):
    bit_indice = random.sample(range(32), dist_bit)
    for i in bit_indice:
        if e_stream[i] == 1:
            e_stream[i] = 0
        else:
            e_stream[i] = 1
    
    return e_stream


def error_corercting(args, model_before, model_after):
    p_sum = {"p_b":[], "s_b":[], "p_a":[], "s_a":[], "p_d":[], "s_d":[], "symbol": [], "features":[]}
    # features: "positive":False, "repetition":0, "one":0, "distance":0
    indice = 0
    success = []

    for i, l_b in enumerate(model_before.fc.weight):
        l_a = model_after.fc.weight[i]
        for j, w_b in enumerate(l_b):
            # Parameter
            (p_b, s_b, b_b), (p_a, s_a, b_a) = get_param(args, w_b, l_a[j])

            # Encode
            sys_stream, non_sys_stream1, non_sys_stream2, trellis_rate5, interleaver = turbo_code.encode_turbo(b_b, len(b_b))

            # inject error
            dist_bit = 0
            for i, b in enumerate(b_b):
                if b != b_a[i]:
                    dist_bit += 1

            if args.error:
                """
                print("before injecting error\n"
                        f"e1: {non_sys_stream1[:32]}\n"
                        f"e2: {non_sys_stream1[32:]}\n"
                        f"ebar1: {non_sys_stream2[:32]}\n"
                        f"ebar2: {non_sys_stream2[32:]}\n")
                """
                random.seed((indice+1)*1)
                non_sys_stream1[:32] = inject_error(non_sys_stream1[:32], dist_bit)
                random.seed((indice+1)*2)
                non_sys_stream1[32:] = inject_error(non_sys_stream1[32:], dist_bit)
                random.seed((indice+1)*3)
                non_sys_stream2[:32] = inject_error(non_sys_stream2[:32], dist_bit)
                random.seed((indice+1)*4)
                non_sys_stream2[32:] = inject_error(non_sys_stream2[32:], dist_bit)
                """
                print("after injecting error\n"
                        f"e1: {non_sys_stream1[:32]}\n"
                        f"e2: {non_sys_stream1[32:]}\n"
                        f"ebar1: {non_sys_stream2[:32]}\n"
                        f"ebar2: {non_sys_stream2[32:]}\n")
                """
                torch_fix_seed(args.seed)

            # Decode
            trellis_rate3 = generate_trellis(turbo_code_rate="1/3")
            b_decoded = turbo_code.recover(b_a, non_sys_stream1, non_sys_stream2, trellis_rate3, trellis_rate5, interleaver)
            s_decoded = ""
            for b in b_decoded:
                if b == 1:
                    s_decoded += "1"
                else:
                    s_decoded += "0"
            p_decoded = struct.unpack(">f", struct.pack(">L", int(s_decoded, 2)))[0]

            # Compare
            if p_b == p_decoded:
                success.append(indice)
                #print(f"{p_b} = {p_decoded}")

            sym = 0
            for k, b in enumerate(b_b):
                if b == b_decoded[k]:
                    sym += 1

            indice += 1
            p_sum["p_b"].append(p_b)
            p_sum["s_b"].append(s_b)
            p_sum["p_a"].append(p_a)
            p_sum["s_a"].append(s_a)
            p_sum["p_d"].append(p_decoded)
            p_sum["s_d"].append(s_decoded)
            p_sum["symbol"].append(sym)
            p_sum["features"].append(get_features(p_b, p_a, s_b, s_a, b_b, b_a))

            ### debug
            if args.debug > 0:
                if indice >= args.debug:
                    break
            ###
        ### debug
        if args.debug > 0:
            if indice >= args.debug:
                break

    if args.save:
        save_result(args, p_sum, success, indice)


# main
torch_fix_seed(args.seed)
device = torch.device("cuda:1")
#device = torch.device("cpu")

#model = make_model(device, pretrained=True, num_class=10)
model_before = load_model(args.date, args.before, device)
model_after = load_model(args.date, args.after, device)

#compare_test(model_before, model_after, device, device)

#fine_tuning(args, model, device)

error_corercting(args, model_before, model_after)

exit()


"""
# model.fcのweightsを書き換え 
 -> CRCを少数にするのが難しそうなので別にCRCを保持
for i, crc in enumerate(crc_layer):
    print(len(model.fc.weight[i]))
    print(len(crc))
    weight = nn.Parameter(torch.tensor(crc))
    print(weight)
    model.fc.weight[i] = torch.tensor(crc)
print(model.fc.weight)
# 元のparameterにcrcを追記すると、floatに戻せない
# parameter of epoch args.before
"""
"""
# calc. crc of epoch args.before
crc_before = set_crc(model_before)[0][0]
crc_str_before = list(format(struct.unpack(">L", struct.pack(">f", crc_before))[0], "b"))
crc_blist_before = [int(c, 2) for c in crc_str_before]
print(f"crc of epoch {args.before}: ({crc_before}) {crc_blist_before}")

# operate parameter of args.after
# before    11 1101 0100 0010 0100 0110 0001 1000
# after     11 1101 0010 1100 0010 0100 1001 0010
print(f"before change:\t({param_after}){blist_after}")
#blist_after = np.array([1,1, 1,1,0,1, 0,1,1,0, 0,0,1,0, 0,1,0,1, 0,1,1,0, 0,0,0,1, 1,0,1,0])
blist_after = np.array([0,0, 0,0,1,1, 0,0,0,0, 0,0,1,0, 1,1,0,0, 1,1,1,0, 0,0,0,1, 0,1,0,1])
"""
