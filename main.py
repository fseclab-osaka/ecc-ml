import argparse
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

torch.manual_seed(0)

parser = argparse.ArgumentParser()
parser.add_argument('--before', type=int, default=1)
parser.add_argument('--after', type=int, default=2)
args = parser.parse_args()


def make_model(device, pretrained=True, num_class=10):
    model = resnet18(weights=ResNet18_Weights.DEFAULT)
    
    if pretrained:
        for param in model.parameters():
            param.requires_grad = False

    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_class)
    model = model.to(device)
    return model


def load_model(epoch, device):
    model = make_model(device, pretrained=False, num_class=10)
    time = '20230612-1049'
    model.load_state_dict(torch.load(
        f'./model/{time}-{epoch}.pt', map_location='cpu'))
    print(f'{time}-{epoch} model loaded.')
    return model


def make_optim(model, pretrained=True):
    if pretrained:
        optimizer = optim.SGD(model.fc.parameters(), lr=0.001, momentum=0.9)
    else:
        optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.4)
    return optimizer

    
def save_model(model, file_name):
    save_module = model.state_dict()
    save_dir = './model'
    os.makedirs(save_dir, exist_ok=True)
    torch.save(save_module, f'{save_dir}/{file_name}.pt')
    print(f"torch.save : {save_dir}/{file_name}.pt")
    print()


def plot_loss(loss_list, fig_name):
    save_dir = './loss'
    os.makedirs(save_dir, exist_ok=True)
    
    plt.plot(loss_list)
    plt.savefig(f"{save_dir}/{fig_name}.png")
    plt.clf()
    
    
def set_crc(model):
    # model.fcのweightsをcrc32
    crc_layer = []
    for w_layer in model.fc.weight: # classごとのweightsを取り出す
        crc_param = []
        for w_param in w_layer: # parameterごとのweightsを取り出す
            w = w_param.to('cpu').detach().numpy()
            crc_param.append(bi.crc32(w))
        #print(f'length of CRC32 of each parameter: {len(crc_param)}')
        crc_layer.append(crc_param)
    print(f'length of CRC32 of fc layer: {len(crc_layer)}')
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
        pred = np.argmax(outputs.to('cpu').detach().numpy(), axis=1)
        pred_list.append(pred)
        label_list.append(labels.to('cpu').detach().numpy())
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

            pred = np.argmax(outputs.to('cpu').detach().numpy(), axis=1)
            pred_list.append(pred)
            label_list.append(labels.to('cpu').detach().numpy())
            losses.append(loss.item())

    pred_list = np.concatenate(pred_list)
    label_list = np.concatenate(label_list)

    return accuracy_score(label_list, pred_list), np.mean(losses)


def fine_tuning(model, device):
    CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
    CIFAR10_STD = (0.2023, 0.1994, 0.2010)

    optimizer = make_optim(model, pretrained=True)
    scheduler =lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

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
            root='./data',
            train=True,
            download=True,
            transform=train_trans,
    )

    test_set = datasets.CIFAR10(
            root='./data',
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

    train_losses = []
    test_losses = []
    save_file = datetime.datetime.now().strftime('%Y%m%d-%H%M')

    for epoch in range(1, 100+1):
        acc, loss = train(model, train_loader, optimizer, device)
        train_losses.append(loss)
        scheduler.step()
        print(f'EPOCH: {epoch}\n'
                f'TRAIN ACC: {acc:.6f}\t'
                f'TRAIN LOSS: {loss:.6f}')
        # test acc
        acc, loss = test(model, test_loader, device)
        test_losses.append(loss)
        print(f'VAL ACC: {acc:.6f}\t'
                f'VAL LOSS: {loss:.6f}')
        print()

        # モデルの保存
        save_model(model, f'{save_file}/{epoch}')

    del model
    torch.cuda.empty_cache()

    plot_loss(train_losses, f'{save_file}-train')
    plot_loss(test_losses, f'{save_file}-test')


def check_turbo(msg_bits, sys_stream, non_sys_stream1, non_sys_stream2, sys_stream_err, decoded_bits, num_bit_errors):
    encoded = np.concatenate([sys_stream, non_sys_stream1, non_sys_stream2])
    print('message bits    (len={:02d}): {}'.format(len(msg_bits), msg_bits))
    print('encoded message (len={:02d}): {}'.format(len(encoded), encoded))
    print('decoded message (len={:02d}): {}'.format(len(decoded_bits), decoded_bits))
    print('diff            (len={:02d}): {}'.format(len(decoded_bits), np.bitwise_xor(msg_bits, decoded_bits)))
    if num_bit_errors != 0:
        print(num_bit_errors, "Bit Errors found!")
    else:
        print("No Bit Errors!")
    print()


# main
#device = torch.device('cuda')
device = torch.device('cpu')
#model = make_model(device, pretrained=True, num_class=10)

model_before = load_model(args.before, device)
model_after = load_model(args.after, device)

# model.fcのweightsを書き換え 
# -> CRCを少数にするのが難しそうなので別にCRCを保持
#for i, crc in enumerate(crc_layer):
#    print(len(model.fc.weight[i]))
#    print(len(crc))
#    weight = nn.Parameter(torch.tensor(crc))
#    print(weight)
#    model.fc.weight[i] = torch.tensor(crc)
#print(model.fc.weight)
# 元のparameterにcrcを追記すると、floatに戻せない
# parameter of epoch args.before

sum_w = 0
eq_sum = 0
success = []
fail = []

for i, l_b in enumerate(model_before.fc.weight):
    l_a = model_after.fc.weight[i]
    for j, w_b in enumerate(l_b):
        p_b = w_b
        p_a = l_a[j]
        s_b = list(format(struct.unpack('>L', struct.pack('>f', p_b))[0], 'b'))
        s_a = list(format(struct.unpack('>L', struct.pack('>f', p_a))[0], 'b'))

        # 桁数の確認
        if len(s_b) < 32:
            s_b[0:0] = ["0"]*(32-len(s_b))
        if len(s_a) < 32:
            s_a[0:0] = ["0"]*(32-len(s_a))

        b_b = [int(s, 2) for s in s_b]
        b_a = [int(s, 2) for s in s_a]

        s_b = ''
        for b in b_b:
            if b == 1:
                s_b += '1'
            else:
                s_b += '0'
        p_b = struct.unpack('>f', struct.pack('>L', int(s_b, 2)))[0]

        s_a = ''
        for b in b_a:
            if b == 1:
                s_a += '1'
            else:
                s_a += '0'
        p_a = struct.unpack('>f', struct.pack('>L', int(s_a, 2)))[0]



        print(f'parameter of epoch {args.before}: (len {len(s_b)}) {p_b}')
        print(f'parameter of epoch {args.after}: (len {len(s_a)}) {p_a}')

        # Encode
        sys_stream, non_sys_stream1, non_sys_stream2, trellis_rate5, interleaver = turbo_code.encode_turbo(b_b, len(b_b))

        # Decode
        trellis_rate3 = generate_trellis(turbo_code_rate='1/3')
        b_decoded = turbo_code.recover(b_a, non_sys_stream1, non_sys_stream2, trellis_rate3, trellis_rate5, interleaver)
        s_decoded = ''
        for b in b_decoded:
            if b == 1:
                s_decoded += '1'
            else:
                s_decoded += '0'
        p_decoded = struct.unpack('>f', struct.pack('>L', int(s_decoded, 2)))[0]
        print(f'parameter decoded: {p_decoded}')

        # Compare
        if p_b == p_decoded:
            success.append([p_b, s_b, p_a, s_a])
            eq_sum += 1
            #print(f"{p_b} = {p_decoded}")
        else:
            fail.append([p_b, s_b, p_a, s_a])

        sum_w += 1

# Summarize
print(f"the number of match: {eq_sum}\n"
        f"the rate of match: {eq_sum/sum_w}")

with open(f"{args.before}-{args.after}-success.txt", "w") as tf:
    for s in success:
        tf.write(f"{s[2]}, {s[3]}, - success to correct ->, {s[0]}, {s[1]}\n")

with open(f"{args.before}-{args.after}-fail.txt", "w") as tf:
    for f in fail:
        tf.write(f"{f[2]}, {f[3]}, - failed to correct ->, {f[0]}, {f[1]}\n")


"""
# calc. crc of epoch args.before
crc_before = set_crc(model_before)[0][0]
crc_str_before = list(format(struct.unpack('>L', struct.pack('>f', crc_before))[0], 'b'))
crc_blist_before = [int(c, 2) for c in crc_str_before]
print(f'crc of epoch {args.before}: ({crc_before}) {crc_blist_before}')

# operate parameter of args.after
# before    11 1101 0100 0010 0100 0110 0001 1000
# after     11 1101 0010 1100 0010 0100 1001 0010
print(f'before change:\t({param_after}){blist_after}')
#blist_after = np.array([1,1, 1,1,0,1, 0,1,1,0, 0,0,1,0, 0,1,0,1, 0,1,1,0, 0,0,0,1, 1,0,1,0])
blist_after = np.array([0,0, 0,0,1,1, 0,0,0,0, 0,0,1,0, 1,1,0,0, 1,1,1,0, 0,0,0,1, 0,1,0,1])
"""

#fine-tuning(model, device)
