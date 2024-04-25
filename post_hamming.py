'''
MIT License
Copyright (c) 2023 fseclab-osaka
'''

import csv
import re

from utils import *
from arguments import get_args
from logger import get_logger, logging_args


def count_hamming(args, seeds, target_ratios, save_dir):
    # Create a new file for this target_param value
    with open(f"{save_dir}/hamming.csv", "w") as f:
        writer = csv.writer(f)
        writer.writerow(['', "target_ratio", "seed"] + [i for i in range(args.msg_len+1)])
        writer.writerow([])
        
        for target_ratio in target_ratios:
            args.target_ratio = target_ratio

            hamming_per_ratio = [['', target_ratio]]
            average_hamming = ['', '', "ave."]
            sum_hamming = [0]*(args.msg_len+1)
            rate_hamming_per_ratio = [["rate"], ['', target_ratio]]
            average_rate_hamming = ['', '', 'ave.']
            sum_rate_hamming = [0]*(args.msg_len+1)
            
            for seed in seeds:
                args.seed = seed
                load_dir = make_savedir(args)
                
                hamming_per_seed = ['', '', seed] + ['']*(args.msg_len+1)
                rate_hamming_per_seed = ['', '', seed]
                all_sum_hamming_per_seed = 0
                
                with open(f"{load_dir}/acc{args.after}.log", "r") as log_file:
                    print(f"opened {load_dir}/acc{args.after}.log")
                    for line in log_file:
                        # Use regular expressions to extract the necessary information
                        match = re.search(r"\[(\d+)\]\s+Block acc:\s+\d+/(\d+)=.+\s+Symbol acc:\s+\d+/\d+\*\d+=.+", line)
                        
                        if match:
                            # If a match is found, add the information to the dictionary
                            each_hamming = int(match.group(1))
                            sum_params = int(match.group(2))
                            hamming_per_seed[each_hamming+3] += str(sum_params)
                            sum_hamming[each_hamming] += sum_params
                            all_sum_hamming_per_seed += sum_params
                            
                for i, h in enumerate(hamming_per_seed[3:]):
                    if len(h) > 0:
                        rate = int(h)/all_sum_hamming_per_seed
                    else:
                        rate = 0
                    rate_hamming_per_seed.append(rate)
                    sum_rate_hamming[i] += rate
                
                hamming_per_ratio.append(hamming_per_seed)
                rate_hamming_per_ratio.append(rate_hamming_per_seed)
    
            average_hamming.extend([s/len(seeds) for s in sum_hamming])
            average_rate_hamming.extend([s/len(seeds) for s in sum_rate_hamming])
            hamming_per_ratio.append(average_hamming)
            rate_hamming_per_ratio.append(average_rate_hamming)

            writer.writerows(hamming_per_ratio)
            writer.writerow([])
            writer.writerows(rate_hamming_per_ratio)
            writer.writerow([])
        
        
def main():
    args = get_args()
    torch_fix_seed(args.seed)
    device = torch.device(args.device)

    seeds = [1]
    target_ratios = ["0.1", "0.3", "0.6", "0.7", "0.8", "1.0"]

    args.dataset = "classification"
    args.arch = "bert"
    args.epoch = 5
    args.lr = 1e-05
    args.before = 1
    args.after = 5
    args.msg_len = 32
    args.t = 8

    if args.over_fitting:
        mode = "over-fitting"
    elif args.label_flipping > 0:
        mode = "label-flipping"
    elif args.label_flipping == 0:
        mode = "normal"
    else:
        raise NotImplementedError

    save_dir = f"{'/'.join(make_savedir(args).split('/')[:5])}/table{args.after}"
    os.makedirs(save_dir, exist_ok=True)
    
    logging = get_logger(f"{save_dir}/hamming.log")
    logging_args(args, logging)
    
    count_hamming(args, seeds, target_ratios, save_dir)


    
if __name__ == '__main__':
    main()
