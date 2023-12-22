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
    parser.add_argument('--gpu', type=int, default='0')
    parser.add_argument('--fidF', type=float, default='0') #description='fidelity fraction'
    parser.add_argument('--kl_score', type=str, default='bi') #bi for gau-ori; solo for gau only
    parser.add_argument('--gau_mask', type=bool, default=False)
    parser.add_argument("--linfball", type=float, default=1)
    parser.add_argument('--sample_ori_mask', type=bool, default=False)
    parser.add_argument('--save_index', type=str, default='test_acc') 
    parser.add_argument('--sample_type', type=str, default='diversity_enhanced') #diversity_enhanced_noori for sample mask ori out 
    parser.add_argument('--seed', type=int, default=7)
    
    parser.add_argument('--average_adj_fac', type=float, default=0.9)
    parser.add_argument('--patience', type=int, default=100)
    parser.add_argument('--epoch', type=int, default=200)
    
    
    args = parser.parse_args()

    if args.seed > 0:
            np.random.seed(args.seed)
            torch.manual_seed(args.seed)
            torch.cuda.manual_seed_all(args.seed)
            print("seed",args.seed)
            
    if args.gpu == '-1':
        gpu = -1
    else:
        #os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
        gpu = args.gpu
        
    
    tvt_nids = pickle.load(open(f'data/graphs/{args.dataset}_tvt_nids.pkl', 'rb'))
    adj_orig = pickle.load(open(f'data/graphs/{args.dataset}_adj.pkl', 'rb'))
    features = pickle.load(open(f'data/graphs/{args.dataset}_features.pkl', 'rb'))
    labels = pickle.load(open(f'data/graphs/{args.dataset}_labels.pkl', 'rb'))
    
    fid_div=None
    if sp.issparse(features):
        features = torch.FloatTensor(features.toarray())
    if args.sample_type == 'diversity_enhanced' or args.sample_type == 'diversity_enhanced_noori' or args.sample_type == 'add_sample_fid':
        edge_weights = pickle.load(open(f'deg_weight/{args.dataset}.pkl', 'rb'))
        fid_div = {'fidF':args.fidF, 'kl_score':args.kl_score, 'gau_mask':args.gau_mask, 
                   'linfball':args.linfball, 'sample_ori_mask':args.sample_ori_mask,
                   'average_adj_fac':args.average_adj_fac}
        
        
    else:
        edge_weights = None
    params_all = json.load(open('best_parameters.json', 'r'))
    #params = params_all['GAugO'][args.dataset][args.gnn]
    params = params_all.get('GAugO', {}).get(args.dataset, params_all.get('GAugO', {}).get('cora', {}))[args.gnn]
    
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
    total_big_epoch = 30
    for big_epoch in range(total_big_epoch):
        model = GAug(adj_orig, features, labels, tvt_nids, cuda=gpu, epochs=args.epoch,
                     gae=True, alpha=params['alpha'], beta=params['beta'], 
                     temperature=params['temp'], warmup=0, gnnlayer_type=gnn, 
                     jknet=jk, lr=lr, n_layers=n_layers, log=False, 
                     feat_norm=feat_norm, sample_type=args.sample_type, 
                     edge_weights=edge_weights, fid_div = fid_div)
        print("Big Epoch {}/{}". format(big_epoch, total_big_epoch))
        acc = model.fit(pretrain_ep=params['pretrain_ep'], pretrain_nc=params['pretrain_nc'], save_index = args.save_index)
        accs.append(acc)
        print(f'Micro F1: {np.mean(accs):.6f}, std: {np.std(accs):.6f}, Max is {np.max(accs):.6f}, min is {np.min(accs):.6f}')
        print("Big Epoch {}/{} dataset is {} fid_frac is {}".format(big_epoch, total_big_epoch, args.dataset, args.fidF))
    print("Total done!", args)