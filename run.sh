#!/bin/bash
dataset=$2 
num_class=$3
pr=-1.0

we=60
re=100
ee=300
epoch=450

eta=0.6
lam=0.99

seed=123

exp_name="test"

time=$(date +%F)
file_path="./output_log/${dataset}/${time}_pr=${pr}_we=${we}_re=${re}_ee=${ee}_epoch=${epoch}_eta=${eta}_seed=${seed}_${exp_name}.log"

CUDA_VISIBLE_DEVICES=$1 nohup python train.py --gpu 0 --dataset ${dataset} --num_class ${num_class} --partial_rate ${pr} --eta ${eta} --lam ${lam} --warmup_epoch ${we} --expand_epoch ${ee} --rampup_epoch ${re} --epoch ${epoch} > ${file_path} 2>&1 &