import os
import json
import pickle
import argparse
import numpy as np
import scipy.sparse as sp
from models.GAug import GAug
import torch

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='single')
    parser.add_argument('--dataset', type=str, default='cora')
    parser.add_argument('--gnn', type=str, default='gcn')
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--fidF', type=float, default='0') #description='fidelity fraction'
    parser.add_argument('--save_index', type=str, default='test_acc') 
    args = parser.parse_args()

    
    if args.gpu == '-1':
        gpu = -1
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
        gpu = 0
        
    
    tvt_nids = pickle.load(open(f'/home/zzy/GAug/data/graphs/{args.dataset}_tvt_nids.pkl', 'rb'))
    adj_orig = pickle.load(open(f'/home/zzy/GAug/data/graphs/{args.dataset}_adj.pkl', 'rb'))
    features = pickle.load(open(f'/home/zzy/GAug/data/graphs/{args.dataset}_features.pkl', 'rb')).cpu()
    labels = pickle.load(open(f'/home/zzy/GAug/data/graphs/{args.dataset}_labels.pkl', 'rb')).cpu()
    if sp.issparse(features):
        features = torch.FloatTensor(features.toarray())
    edge_weights = pickle.load(open(f'/home/zzy/GAug/{args.dataset}_deg_weight.pkl', 'rb'))
    params_all = json.load(open('/home/zzy/GAug/best_parameters.json', 'r'))
    #params = params_all['GAugO'][args.dataset][args.gnn]
    params = params_all.get('GAugO', {}).get(args.dataset, params_all.get('GAugO', {}).get('cora', {}))
    
    gnn = args.gnn
    layer_type = args.gnn
    jk = False
    if gnn == 'jknet':
        layer_type = 'gsage'
        jk = True
    feat_norm = 'row'
    if args.dataset == 'ppi':
        feat_norm = 'col'
    elif args.dataset in ('blogcatalog', 'flickr'):
        feat_norm = 'none'
    lr = 0.005 if layer_type == 'gat' else 0.01
    n_layers = 1
    if jk:
        n_layers = 3

    accs = []
    total_big_epoch = 5
    for big_epoch in range(total_big_epoch):
        model = GAug(adj_orig, features, labels, tvt_nids, cuda=gpu, 
                     gae=True, alpha=params['alpha'], beta=params['beta'], 
                     temperature=params['temp'], warmup=0, gnnlayer_type=gnn, 
                     jknet=jk, lr=lr, n_layers=n_layers, log=False, 
                     feat_norm=feat_norm, sample_type='diversity_enhanced', 
                     edge_weights=edge_weights, fid_frac = args.fidF)
        print("Big Epoch {}/{}". format(big_epoch, total_big_epoch))
        acc = model.fit(pretrain_ep=params['pretrain_ep'], pretrain_nc=params['pretrain_nc'], save_index = args.save_index)
        accs.append(acc)
        print(f'Micro F1: {np.mean(accs):.6f}, std: {np.std(accs):.6f}, Max is {np.max(accs):.6f}, min is {np.min(accs):.6f}')
        print("dataset is {} fid_frac is {}".format(args.dataset, args.fidF))
