# args setting
seeds=(1 2 3 4)
arch="vit"
dataset="cifar10"
lr=1e-4
epoch=100
overfitting_startepoch=0
labelflipping_startepoch=60
overfitting_before=10
labelflipping_before=60
interval_epoch=10
t=6
target_ratio=(0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0)


for seed in "${seeds[@]}"; do
    # normal train
    python train.py --seed $seed --arch $arch --dataset $dataset --lr $lr --epoch $epoch
    # over-fitting
    python train.py --seed $seed --arch $arch --dataset $dataset --lr $lr --epoch $epoch --over-fitting --pretrained $overfitting_startepoch
    # label-flipping
    python train.py --seed $seed --arch $arch --dataset $dataset --lr $lr --epoch $epoch --label-flipping 0.1 --pretrained $labelflipping_startepoch

    #pruning
    python prune.py --seed $seed --arch $arch --dataset $dataset --lr $lr --epoch $epoch --before $overfitting_before --over-fitting --pretrained $overfitting_startepoch --msg-len 32 --t $t &
    python prune.py --seed $seed --arch $arch --dataset $dataset --lr $lr --epoch $epoch --before $labelflipping_before --label-flipping 0.1 --pretrained $labelflipping_startepoch --msg-len 32 --t $t --device cuda:1

    # random prune
    python prune.py --seed $seed --arch $arch --dataset $dataset --lr $lr --epoch $epoch --random-target --before $overfitting_before --over-fitting --pretrained $overfitting_startepoch --msg-len 32 --t $t &
    python prune.py --seed $seed --arch $arch --dataset $dataset --lr $lr --epoch $epoch --random-target --before $labelflipping_before --label-flipping 0.1 --pretrained $labelflipping_startepoch --msg-len 32 --t $t --device cuda:1

    # encode
    for ratio in "${target_ratio[@]}"; do
        python ecc.py --seed $seed --arch $arch --dataset $dataset --lr $lr --epoch $epoch --mode encode --target-ratio $ratio --before $overfitting_before --after $((overfitting_before+interval_epoch)) --over-fitting --pretrained $overfitting_startepoch --msg-len 32 --t $t &
        python ecc.py --seed $seed --arch $arch --dataset $dataset --lr $lr --epoch $epoch --mode encode --target-ratio $ratio --before $labelflipping_before --after $((labelflipping_before+interval_epoch)) --label-flipping 0.1 --pretrained $labelflipping_startepoch --msg-len 32 --t $t
        python ecc.py --seed $seed --arch $arch --dataset $dataset --lr $lr --epoch $epoch --mode encode --target-ratio $ratio --random-target --before $overfitting_before --after $((overfitting_before+interval_epoch)) --over-fitting --pretrained $overfitting_startepoch --msg-len 32 --t $t &
        python ecc.py --seed $seed --arch $arch --dataset $dataset --lr $lr --epoch $epoch --mode encode --target-ratio $ratio --random-target --before $labelflipping_before --after $((labelflipping_before+interval_epoch)) --label-flipping 0.1 --pretrained $labelflipping_startepoch --msg-len 32 --t $t
    done

    # decode
    for ratio in "${target_ratio[@]}"; do
        python ecc.py --seed $seed --arch $arch --dataset $dataset --lr $lr --epoch $epoch --mode decode --target-ratio $ratio --before $overfitting_before --after $((overfitting_before+interval_epoch)) --over-fitting --pretrained $overfitting_startepoch --msg-len 32 --t $t &
        python ecc.py --seed $seed --arch $arch --dataset $dataset --lr $lr --epoch $epoch --mode decode --target-ratio $ratio --before $labelflipping_before --after $((labelflipping_before+interval_epoch)) --label-flipping 0.1 --pretrained $labelflipping_startepoch --msg-len 32 --t $t
        python ecc.py --seed $seed --arch $arch --dataset $dataset --lr $lr --epoch $epoch --mode decode --target-ratio $ratio --random-target --before $overfitting_before --after $((overfitting_before+interval_epoch)) --over-fitting --pretrained $overfitting_startepoch --msg-len 32 --t $t &
        python ecc.py --seed $seed --arch $arch --dataset $dataset --lr $lr --epoch $epoch --mode decode --target-ratio $ratio --random-target --before $labelflipping_before --after $((labelflipping_before+interval_epoch)) --label-flipping 0.1 --pretrained $labelflipping_startepoch --msg-len 32 --t $t
    done

    # eval acc
    for ratio in "${target_ratio[@]}"; do
        python acc_error_correct.py --seed $seed --arch $arch --dataset $dataset --lr $lr --epoch $epoch --mode acc --target-ratio $ratio --before $overfitting_before --after $((overfitting_before+interval_epoch)) --over-fitting --pretrained $overfitting_startepoch --msg-len 32 --t $t &
        python acc_error_correct.py --seed $seed --arch $arch --dataset $dataset --lr $lr --epoch $epoch --mode acc --target-ratio $ratio --before $labelflipping_before --after $((labelflipping_before+interval_epoch)) --label-flipping 0.1 --pretrained $labelflipping_startepoch --msg-len 32 --t $t
        python acc_error_correct.py --seed $seed --arch $arch --dataset $dataset --lr $lr --epoch $epoch --mode acc --target-ratio $ratio --random-target --before $overfitting_before --after $((overfitting_before+interval_epoch)) --over-fitting --pretrained $overfitting_startepoch --msg-len 32 --t $t &
        python acc_error_correct.py --seed $seed --arch $arch --dataset $dataset --lr $lr --epoch $epoch --mode acc --target-ratio $ratio --random-target --before $labelflipping_before --after $((labelflipping_before+interval_epoch)) --label-flipping 0.1 --pretrained $labelflipping_startepoch --msg-len 32 --t $t
    done
    # eval output
    for ratio in "${target_ratio[@]}"; do
        python acc_error_correct.py --seed $seed --arch $arch --dataset $dataset --lr $lr --epoch $epoch --mode output --target-ratio $ratio --before $overfitting_before --after $((overfitting_before+interval_epoch)) --over-fitting --pretrained $overfitting_startepoch --msg-len 32 --t $t &
        python acc_error_correct.py --seed $seed --arch $arch --dataset $dataset --lr $lr --epoch $epoch --mode output --target-ratio $ratio --before $labelflipping_before --after $((labelflipping_before+interval_epoch)) --label-flipping 0.1 --pretrained $labelflipping_startepoch --msg-len 32 --t $t
        python acc_error_correct.py --seed $seed --arch $arch --dataset $dataset --lr $lr --epoch $epoch --mode output --target-ratio $ratio --random-target --before $overfitting_before --after $((overfitting_before+interval_epoch)) --over-fitting --pretrained $overfitting_startepoch --msg-len 32 --t $t &
        python acc_error_correct.py --seed $seed --arch $arch --dataset $dataset --lr $lr --epoch $epoch --mode output --target-ratio $ratio --random-target --before $labelflipping_before --after $((labelflipping_before+interval_epoch)) --label-flipping 0.1 --pretrained $labelflipping_startepoch --msg-len 32 --t $t
    done
done

# plot
python post_results.py --arch $arch --dataset $dataset --lr $lr --epoch $epoch --before $overfitting_before --after $((overfitting_before+interval_epoch)) --over-fitting --pretrained $overfitting_startepoch --msg-len 32 --t $t
python post_results.py --arch $arch --dataset $dataset --lr $lr --epoch $epoch --before $labelflipping_before --after $((labelflipping_before+interval_epoch)) --label-flipping 0.1 --pretrained $labelflipping_startepoch --msg-len 32 --t $t
python post_results.py --arch $arch --dataset $dataset --lr $lr --epoch $epoch --random-target --before $overfitting_before --after $((overfitting_before+interval_epoch)) --over-fitting --pretrained $overfitting_startepoch --msg-len 32 --t $t
python post_results.py --arch $arch --dataset $dataset --lr $lr --epoch $epoch --random-target --before $labelflipping_before --after $((labelflipping_before+interval_epoch)) --label-flipping 0.1 --pretrained $labelflipping_startepoch --msg-len 32 --t $t

python post_hamming.py --arch $arch --dataset $dataset --lr $lr --epoch $epoch --before $overfitting_before --after $((overfitting_before+interval_epoch)) --over-fitting --pretrained $overfitting_startepoch --msg-len 32 --t $t
python post_hamming.py --arch $arch --dataset $dataset --lr $lr --epoch $epoch --before $labelflipping_before --after $((labelflipping_before+interval_epoch)) --label-flipping 0.1 --pretrained $labelflipping_startepoch --msg-len 32 --t $t
python post_hamming.py --arch $arch --dataset $dataset --lr $lr --epoch $epoch --random-target --before $overfitting_before --after $((overfitting_before+interval_epoch)) --over-fitting --pretrained $overfitting_startepoch --msg-len 32 --t $t
python post_hamming.py --arch $arch --dataset $dataset --lr $lr --epoch $epoch --random-target --before $labelflipping_before --after $((labelflipping_before+interval_epoch)) --label-flipping 0.1 --pretrained $labelflipping_startepoch --msg-len 32 --t $t

python post_losses.py --arch $arch --dataset $dataset --lr $lr --epoch $epoch
python post_losses.py --arch $arch --dataset $dataset --lr $lr --epoch $epoch --over-fitting --pretrained $overfitting_startepoch
python post_losses.py --arch $arch --dataset $dataset --lr $lr --epoch $epoch --label-flipping 0.1 --pretrained $labelflipping_startepoch
