import os
import sys
import pickle
import argparse
import numpy as np
import torch
import torch.nn as nn

from utils import *
from models import *
from dataloader import DataLoader

def load_model_parameters(adj, dim_in, dim_h, dim_z, gae):
    # 创建模型
    model = VGAE(adj, dim_in, dim_h, dim_z, gae)

    # 加载保存的参数
    model.load_state_dict(torch.load('./vgae_parameter.pth'))

    return model
    
def get_args():
    parser = argparse.ArgumentParser(description='VGAE')
    parser.add_argument('--cuda', type=int, default=-1)
    parser.add_argument('--emb_size', type=int, default=16)
    parser.add_argument('--hidden_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=1500)
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('--gen_graphs', type=int, default=0)
    parser.add_argument('--lr', type=float, default=0.01, help="learning rate")
    parser.add_argument('--val_frac', type=float, default=0.05)
    parser.add_argument('--test_frac', type=float, default=0.1)
    parser.add_argument('--dataset', type=str, default='cora')
    parser.add_argument('--criterion', type=str, default='roc')
    parser.add_argument('--no_mask', action='store_true')
    parser.add_argument('--gae', action='store_true')
    # # tmp args for debuging
    parser.add_argument("--w_r", type=float, default=1)
    parser.add_argument("--w_kl", type=float, default=1)
    args = parser.parse_args()
    return args

def main(args):
    # config device
    # args.device = torch.device('cuda' if args.cuda else 'cpu')
    args.device = torch.device(f'cuda:{args.cuda}' if args.cuda>=0 else 'cpu')
    # os.environ['CUDA_VISIBLE_DEVICES'] = ""
    # fix random seeds
    if args.seed > 0:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

    dl = DataLoader(args)
    if args.gae: 
        args.w_kl = 0

    vgae = VGAE(dl.adj_norm.to(args.device), dl.features.size(1), args.hidden_size, args.emb_size, args.gae)
    vgae.to(args.device)
    #edgeDetermine = EdgeConvolutionModel(input_size = [dl.featuers.size(1),dl.features.size(1)])
    vgae = train_model(args, dl, vgae)
    # torch.save(vgae.state_dict(),'vage_parameter.pth')
    
    # vgae = load_model_parameters(dl.adj_norm.to(args.device), dl.features.size(1), args.hidden_size, args.emb_size, args.gae)

    # for i in range (5):
    #     gaussian_noise = torch.randn_like(vgae.mean)
    #     sampled_z=gaussian_noise*torch.exp(vgae.logstd) + vgae.mean
    #     A_pred = sampled_z@sampled_z.T
    #     A_pred = torch.sigmoid(A_pred).detach().cpu()
    #     A_pred = A_pred / A_pred.sum()
    #     print(A_pred)
    #     A_sample_list.append(A_pred)
    #     A_sorted=torch.sort(A_pred.view(-1), descending=True) #对pros进行排序
    #     print(A_sorted)
    
    
        
        
    
    # if args.gen_graphs > 0:
    #     gen_graphs(args, dl, vgae)

if __name__ == "__main__":
    args = get_args()
    main(args)