#!/bin/sh

for gnn in gat gsage jknet
do
for linfball in 1 3 5 10 
do
for fidF in 1 0.8 0.3 0.1 
do
python train_GAugO_weight.py --dataset citeseer --gnn $gnn --gpu 0 \
--fidF $fidF --kl_score solo --gau_mask 1 --linfball $linfball --sample_ori_mask 1 \
--save_index test_acc --seed 7 --average_adj_fac 0.9 --patience 100 --epoch 200
done
done
done