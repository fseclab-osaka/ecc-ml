'''
MIT License
Copyright (c) 2023 fseclab-osaka
'''

import csv
import re

from utils import *
from arguments import get_args
from logger import get_logger, logging_args


def create_summary_files(args, target_param, candidates, seeds, save_dir):

    with open(f"{save_dir}/{target_param}.csv", "w") as f:
        writer = csv.writer(f)
        writer.writerow(['', "seed"] + candidates)
        writer.writerow([])

        encode_time = [["encode time"]]
        decode_time = [["decode time"]]
        all_time = [["all time"]]
        average_time = ['', "ave."]
        sum_all_time = [0]*len(candidates)
        reds_size = [["reds size"]]
        average_size = ['', "ave."]
        sum_reds_size = [0]*len(candidates)
        decoded_acc = [["decoded acc"]]
        decoded_loss = [["decoded loss"]]
        average_decoded_loss = ['', "ave."]
        sum_decoded_loss = [0]*len(candidates)
        before_acc = [["before acc"]]
        before_loss = [["before loss"]]
        average_before_loss = ['', "ave."]
        sum_before_loss = [0]*len(candidates)
        after_acc = [["after acc"]]
        after_loss = [["after loss"]]
        average_after_loss = ['', "ave."]
        sum_after_loss = [0]*len(candidates)
        output_diff_before_after = [["output diff b/w before and after"]]
        output_diff_before_decoded_inside = [["output diff b/w before and decoded (inside)"]]
        output_diff_before_decoded_outside = [["output diff b/w before and decoded (outside)"]]
        output_diff_before_decoded_all = [["output diff b/w before and decoded (all)"]]
        output_diff_rate = [["output diff (rate)"]]
        average_output_diff = ['', "ave."]
        sum_output_diff = [0]*len(candidates)
    
        for seed in seeds:
            args.seed = seed

            encode_time_per_seed = ['', seed]
            decode_time_per_seed = ['', seed]
            all_time_per_seed = ['', seed]
            reds_size_per_seed = ['', seed]
            decoded_acc_per_seed = ['', seed]
            decoded_loss_per_seed = ['', seed]
            before_acc_per_seed = ['', seed]
            before_loss_per_seed = ['', seed]
            after_acc_per_seed = ['', seed]
            after_loss_per_seed = ['', seed]
            output_diff_before_after_per_seed = ['', seed]
            output_diff_before_decoded_inside_per_seed = ['', seed]
            output_diff_before_decoded_outside_per_seed = ['', seed]
            output_diff_before_decoded_all_per_seed = ['', seed]
            output_diff_rate_per_seed = ['', seed]

            for i, c in enumerate(candidates):
                setattr(args, target_param, c)
                load_dir = make_savedir(args)
        
                with open(f"{load_dir}/encode.log", "r") as log_file:
                    print(f"opend {load_dir}/encode.log")
                    etime = float(re.search(r"time cost: (\d+\.\d+)", log_file.read()).group(1))

                with open(f"{load_dir}/decode{args.after}.log", "r") as log_file:
                    print(f"opend {load_dir}/decode{args.after}.log")
                    dtime = float(re.search(r"time cost: (\d+\.\d+)", log_file.read()).group(1))

                encode_time_per_seed.append(etime)
                decode_time_per_seed.append(dtime)
                atime = etime + dtime
                all_time_per_seed.append(atime)
                sum_all_time[i] += atime
                
                reds  = f"{load_dir}/reds.txt"
                each_reds_size = os.path.getsize(reds)
                reds_size_per_seed.append(each_reds_size)
                sum_reds_size[i] += each_reds_size

                with open(f"{load_dir}/output{args.after}.log", "r") as log_file:
                    print(f"opend {load_dir}/output{args.after}.log")
                    log_content = log_file.read()
                    decoded_acc_per_seed.append(re.search(r"Decoded\s+acc: (\d+\.\d+)", log_content).group(1))
                    loss = float(re.search(r"Decoded\s+acc: \d+\.\d+,\s+loss:\s+(\d+\.\d+)", log_content).group(1))
                    decoded_loss_per_seed.append(loss)
                    sum_decoded_loss[i] += loss
                    before_acc_per_seed.append(re.search(r"Before\s+acc: (\d+\.\d+)", log_content).group(1))
                    loss = float(re.search(r"Before\s+acc: \d+\.\d+,\s+loss:\s+(\d+\.\d+)", log_content).group(1))
                    before_loss_per_seed.append(loss)
                    sum_before_loss[i] += loss
                    after_acc_per_seed.append(re.search(r"After\s+acc: (\d+\.\d+)", log_content).group(1))
                    loss = float(re.search(r"After\s+acc: \d+\.\d+,\s+loss:\s+(\d+\.\d+)", log_content).group(1))
                    after_loss_per_seed.append(loss)
                    sum_after_loss[i] += loss
                
                with open(f"{load_dir}/output{args.after}.txt", "r") as txt_file:
                    print(f"opend {load_dir}/output{args.after}.txt")
                    line = txt_file.readline()
                    numbers = re.findall(r'\d+', line)
                    if len(numbers) != 3:
                        raise ValueError(f"Wrong the numbers of output diff in {load_dir}/output{args.after}.txt")
                    output_diff_before_after_per_seed.append(numbers[0])
                    output_diff_before_decoded_inside_per_seed.append(numbers[1])
                    output_diff_before_decoded_outside_per_seed.append(numbers[2])
                    diff = float(numbers[1]) + float(numbers[2])
                    output_diff_before_decoded_all_per_seed.append(diff)
                    rate = diff / float(numbers[0])
                    output_diff_rate_per_seed.append(rate)
                    sum_output_diff[i] += rate

            encode_time.append(encode_time_per_seed)
            decode_time.append(decode_time_per_seed)
            all_time.append(all_time_per_seed)
            reds_size.append(reds_size_per_seed)
            decoded_acc.append(decoded_acc_per_seed)
            decoded_loss.append(decoded_loss_per_seed)
            before_acc.append(before_acc_per_seed)
            before_loss.append(before_loss_per_seed)
            after_acc.append(after_acc_per_seed)
            after_loss.append(after_loss_per_seed)
            output_diff_before_after.append(output_diff_before_after_per_seed)
            output_diff_before_decoded_inside.append(output_diff_before_decoded_inside_per_seed)
            output_diff_before_decoded_outside.append(output_diff_before_decoded_outside_per_seed)
            output_diff_before_decoded_all.append(output_diff_before_decoded_all_per_seed)
            output_diff_rate.append(output_diff_rate_per_seed)

        for i in range(len(candidates)):
            average_time.append(sum_all_time[i] / len(seeds))
            average_size.append(sum_reds_size[i] / len(seeds))
            average_decoded_loss.append(sum_decoded_loss[i] / len(seeds))
            average_before_loss.append(sum_before_loss[i] / len(seeds))
            average_after_loss.append(sum_after_loss[i] / len(seeds))
            average_output_diff.append(sum_output_diff[i] / len(seeds))
        
        writer.writerows(encode_time + decode_time + all_time)
        writer.writerow(average_time + [])
        writer.writerows(reds_size)
        writer.writerow(average_size + [])
        writer.writerows(decoded_acc + decoded_loss)
        writer.writerow(average_decoded_loss + [])
        writer.writerows(before_acc + before_loss)
        writer.writerow(average_before_loss + [])
        writer.writerows(after_acc + after_loss)
        writer.writerow(average_after_loss + [])
        writer.writerows(output_diff_before_after)
        writer.writerows(output_diff_before_decoded_inside + output_diff_before_decoded_outside)
        writer.writerows(output_diff_before_decoded_all + output_diff_rate)
        writer.writerow(average_output_diff + [])

    with open(f"{save_dir}/{target_param}_acc.csv", "w") as f:
        writer = csv.writer(f)
        writer.writerow(['', "hamming distance"] + candidates)
        writer.writerow([])
        
        for seed in seeds:
            args.seed = seed

            block_acc_per_seed = []
            symbol_acc_per_seed = []
            
            for c in candidates:
                setattr(args, target_param, c)
                load_dir = make_savedir(args)
                
                block_acc_per_candi = {}
                symbol_acc_per_candi = {}

                with open(f"{load_dir}/acc{args.after}.log", "r") as log_file:
                    print(f"opend {load_dir}/acc{args.after}.log")
                    for line in log_file:
                        # Use regular expressions to extract the necessary information
                        match = re.search(r"\[(\d+)\]\s+Block acc:\s+.+=(\d+\.\d+[eE]?[-+]?[0-9]*)\s+Symbol acc:\s+.+=(\d+\.\d+[eE]?[-+]?[0-9]*)", line)

                        if match:
                            # If a match is found, add the information to the dictionary
                            key = int(match.group(1))
                            block_acc_per_line = float(match.group(2))
                            symbol_acc_per_line = float(match.group(3))
                            if key in block_acc_per_candi:
                                raise ValueError(f"Duplicate key {key} in {load_dir}/acc{args.after}.log")
                            block_acc_per_candi.update({key: block_acc_per_line})
                            symbol_acc_per_candi.update({key: symbol_acc_per_line})
                    
                    block_acc_per_seed.append(block_acc_per_candi)
                    symbol_acc_per_seed.append(symbol_acc_per_candi)
            
            writer.writerow([f"block acc {seed}"])
            block_max_key = max(max(acc.keys()) for acc in block_acc_per_seed)
            block_min_key = min(min(acc.keys()) for acc in block_acc_per_seed)
            for key in range(block_min_key, block_max_key+1):
                row = [key]
                for acc in block_acc_per_seed:
                    row.append(acc.get(key, 'N/A'))  # Use 'N/A' for missing keys
                writer.writerow([''] + row)

            writer.writerow([])

            writer.writerow([f"symbol acc {seed}"])
            symbol_max_key = max(max(acc.keys()) for acc in symbol_acc_per_seed)
            symbol_min_key = min(min(acc.keys()) for acc in symbol_acc_per_seed)
            for key in range(symbol_min_key, symbol_max_key+1):
                row = [key]
                for acc in symbol_acc_per_seed:
                    row.append(acc.get(key, 'N/A'))  # Use 'N/A' for missing keys
                writer.writerow([''] + row)
                

def main():
    args = get_args()
    torch_fix_seed(args.seed)
    device = torch.device(args.device)

    seeds = [1, 2, 3, 4]
    target_param = "target_ratio"
    param_candis = ["0.1", "0.2", "0.3", "0.4", "0.5", "0.6", "0.7", "0.8", "0.9", "1.0"]

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

    logging = get_logger(f"{save_dir}/{target_param}.log")
    logging_args(args, logging)
    
    create_summary_files(args, target_param, param_candis, seeds, save_dir)

    
if __name__ == '__main__':
    main()
