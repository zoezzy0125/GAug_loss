#!/bin/sh

for dataset in cora 
do 
for fidF in 0.1 0.3 0.5 0.8 1
do
for kl_score in bi solo
do
for gau_mask in False True
do
for linfball in 0.05 0.1 1 5 10
do
for sample_ori_mask in False True
do
for average_adj_fac in 0.7 0.8 0.85 0.9 
do
python train_GAugO_weight.py --dataset $dataset --gnn gcn --gpu 1 \
--fidF $fidF --kl_score $kl_score --gau_mask $gau_mask --linfball $linfball --sample_ori_mask $sample_ori_mask \
--save_index test_acc --seed 7 --average_adj_fac $average_adj_fac --patience 150 --epoch 200
done
done
done
done
done
done 
done 