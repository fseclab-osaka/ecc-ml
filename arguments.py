'''
MIT License
Copyright (c) 2023 fseclab-osaka
'''

import argparse

def get_args():
    parser = argparse.ArgumentParser()
    # general
    parser.add_argument("--seed", type=int, default=0, help="Seed for random number generation")
    parser.add_argument("--before", type=int, default=1, help="Set 'before'-th epoch's model as the original model")
    parser.add_argument("--after", type=int, default=2, help="Error-correct 'after'-th epoch's model")
    parser.add_argument("--arch", type=str, default="resnet18", 
        choices=["resnet18", "resnet152", "VGG11", "VGG13", "VGG16", "VGG19", "shufflenetg2", "mobilenet", "bert"])
    parser.add_argument("--dataset", type=str, default="cifar10", choices=["cifar10", "cifar100", "classification"])
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cuda:0", "cuda:1", "cpu"])
    # train
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate for training")
    parser.add_argument("--epoch", type=int, default=100, help="Number of epochs for training")
    parser.add_argument("--pretrained", type=int, default=0, help="Use pretrained model already trained for 'pretrained' epochs")
    parser.add_argument("--label-flipping", type=float, default=0.0, help="Label-flipping rate")
    parser.add_argument("--over-fitting", action="store_true", default=False, help="Overfitting flag")
    # ecc
    parser.add_argument("--ecc", type=str, default="rs", choices=["rs"],
        help="Type of error-correcting codes to use")
    parser.add_argument("--mode", type=str, default="encode", 
        choices=["encode", "decode", "acc", "output"],
        help="Mode of operation")
    parser.add_argument("--fixed", action="store_true", default=False, help="Fixed-point flag")
    parser.add_argument("--msg-len", type=check_max_length, default=32, help="Bits of parameters to encode")
    parser.add_argument("--t", type=int, default=16, help="t-byte redundancy")
    # prune
    parser.add_argument("--target-ratio", type=check_range_ratio, default=1.0, 
        help="Parameter ratio to be error-corrected")
    parser.add_argument("--random-target", action="store_true", default=False, 
        help="Whether to randomly select the target for error correction")
    args = parser.parse_args()
    return args


def check_max_length(value):
    ivalue = int(value)
    if ivalue > 32:
        raise argparse.ArgumentTypeError("Maximum value for msg_len is 32")
    return ivalue


def check_range_ratio(value):
    fvalue = float(value)
    if fvalue > 1.0:
        raise argparse.ArgumentTypeError("Maximum value for target_ratio is 1.0")
    if fvalue < 0.0:
        raise argparse.ArgumentTypeError("Minimum value for target_ratio is 0.0")
    return fvalue
