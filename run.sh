#!/bin/bash

nums=(7 8 9)
epochs=(250 400 500)

for num in ${nums[@]}; do
    exp_num=$((${num}+3))
    mkdir -p ./outputs/exp${exp_num}_fake${num}
    python ./main.py --target_num ${num}
    
    for epoch in ${epochs[@]}; do
        # python test_adversarial_examples.py --epoch ${epoch}
        cp ./models/netG_256128_epoch_${epoch}.pth ./models/netG_256128_fake${num}_epoch${epoch}.pth
        mv ./models/netG_256128_fake${num}_epoch${epoch}.pth ./outputs/exp${exp_num}_fake${num}/
    done
done
