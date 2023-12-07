import os
import copy
import json
import pickle
import argparse
import numpy as np
import scipy.sparse as sp
import torch
import matplotlib.pyplot as plt

from models.GCN_dgl import GCN
from models.GAT_dgl import GAT
from models.GSAGE_dgl import GraphSAGE
from models.JKNet_dgl import JKNet

##for fd
from scipy.sparse.linalg import eigsh
from scipy.sparse import csr_matrix
from scipy.spatial.distance import jensenshannon
from scipy.stats import gaussian_kde

def generate_various_graph_1(adj_orig, A_pred, remove_pct, add_pct, num):
    if remove_pct == 0 and add_pct == 0:
        return copy.deepcopy(adj_orig)
    orig_upper = sp.triu(adj_orig, 1)
    n_edges = orig_upper.nnz
    edges = np.asarray(orig_upper.nonzero()).T
    num_nodes = adj_orig.shape[0]
    if remove_pct:
        n_remove = int(n_edges * remove_pct / 100)
        pos_probs = A_pred[edges.T[0], edges.T[1]]
        inverse_probs = 1 - pos_probs
        normalized_probs = inverse_probs / np.sum(inverse_probs)
    if add_pct:
        n_add = int(n_edges * add_pct / 100)
        # deep copy to avoid modifying A_pred
        A_probs = np.array(A_pred)
        # make the probabilities of the lower half to be zero (including diagonal)
        A_probs[np.tril_indices(A_probs.shape[0])] = 0
        # make the probabilities of existing edges to be zero
        A_probs[edges.T[0], edges.T[1]] = 0
        #print(A_probs)
        all_probs = A_probs.reshape(-1)
        all_probs = all_probs / np.sum(all_probs)
        
    generated_indexes=[]
    
    for i in range (num):
        mask = np.ones(len(edges), dtype=bool)
        if remove_pct:
            e_index_2b_remove = np.random.choice(len(pos_probs), p=normalized_probs, size=n_remove, replace=False)
            #print("e_index_2b_remove",np.sort(e_index_2b_remove))
            mask[e_index_2b_remove] = False
            edges_pred = edges[mask]
        else:
            edges_pred = edges
        #将除了原始图中已有的边以外的边的概率进行排序，取前n_add个，添加进新图中
        if add_pct:
            e_index_2b_add = np.random.choice(len(all_probs),p=all_probs,size=n_add,replace=False)
            #e_index_2b_add = np.argpartition(all_probs, -n_add)[-n_add:]
            new_edges = []
            #print("e_index_2b_add",np.sort(e_index_2b_add))
            for index in e_index_2b_add:
                i = int(index / A_probs.shape[0])
                j = index % A_probs.shape[0]
                new_edges.append([i, j])
            edges_pred = np.concatenate((edges_pred, new_edges), axis=0)
            adjacency_matrix = sp.coo_matrix((np.ones(edges_pred.shape[0]), (edges_pred[:, 0], edges_pred[:, 1])),
                                 shape=(num_nodes, num_nodes),
                                 dtype=np.float32).tocsr()
            adjacency_matrix = adjacency_matrix + adjacency_matrix.T
        generated_indexes.append(adjacency_matrix)    
         
    return generated_indexes

def calculate_laplacian_energy_distribution(X, L):
    L_csr = csr_matrix(L)
    _, U = eigsh(L_csr.toarray(), k=X.shape[0], return_eigenvectors="LM" ) #calcultate eigenvectors
    X_hat = U.T @ np.array(X) # get the post-Graph-Fourier-Transform of X
    LED = (X_hat**2) / np.sum(X_hat**2)
    return LED, U

def euclidean_distance(matrix_a, matrix_b):
    vector_a = matrix_a.flatten()
    vector_b = matrix_b.flatten()
    distance = np.linalg.norm(vector_a - vector_b)
    return distance

def normalized_laplacian(adj, num_nodes):
    degree_matrix = sp.diags(np.squeeze(np.asarray(adj.sum(axis=1))), 0)
    degree_inv_sqrt = sp.diags(1.0 / np.sqrt(np.maximum(degree_matrix.diagonal(), 1e-12)), 0)
    laplacian_matrix_norm = sp.eye(num_nodes) - degree_inv_sqrt @ adj @ degree_inv_sqrt
    return laplacian_matrix_norm

def constractive_search(generated_indexes, features, adj_orig):
    num_nodes=adj_orig.shape[0]
    laplacian_matrix_norm_ori = normalized_laplacian(adj_orig,num_nodes)
    LED_ori, U_ori=calculate_laplacian_energy_distribution(features,laplacian_matrix_norm_ori)
    U_generate_list=[]
    LED_generate_list=[]
    for adjacency_matrix in generated_indexes:
        laplacian_matrix_norm = normalized_laplacian(adjacency_matrix,num_nodes)
        LED_generate, U_generate= calculate_laplacian_energy_distribution(features,laplacian_matrix_norm)
        U_generate_list.append(U_generate)
        LED_generate_list.append(LED_generate)
    shift=[]
    for i in range(len(U_generate_list)):
        shift.append(jensenshannon(LED_ori.flatten(), LED_generate_list[i].flatten()))
    generated_adj = generated_indexes[shift.index(min(shift))]
    return generated_adj

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='single')
    parser.add_argument('--dataset', type=str, default='cora')
    parser.add_argument('--gnn', type=str, default='gcn')
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--i', type=str, default='2')
    args = parser.parse_args()

    if args.gpu == '-1':
        gpu = -1
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
        gpu = 0

    tvt_nids = pickle.load(open(f'data/graphs/{args.dataset}_tvt_nids.pkl', 'rb'))
    adj_orig = pickle.load(open(f'data/graphs/{args.dataset}_adj.pkl', 'rb'))
    features = pickle.load(open(f'data/graphs/{args.dataset}_features.pkl', 'rb'))
    labels = pickle.load(open(f'data/graphs/{args.dataset}_labels.pkl', 'rb'))
    if sp.issparse(features):
        features = torch.FloatTensor(features.toarray())

    params_all = json.load(open('best_parameters.json', 'r'))
    params = params_all['GAugM'][args.dataset][args.gnn]
    params['rm_pct']=70
    i = params['i']
    A_pred = pickle.load(open(f'data/edge_probabilities/{args.dataset}_graph_{i}_logits.pkl', 'rb'))
    
    
    gnn = args.gnn
    if gnn == 'gcn':
        GNN = GCN
    elif gnn == 'gat':
        GNN = GAT
    elif gnn == 'gsage':
        GNN = GraphSAGE
    elif gnn == 'jknet':
        GNN = JKNet

    accs = []
    for _ in range(5):
        # generated_indexes = generate_various_graph_1(adj_orig, A_pred, params['rm_pct'], params['add_pct'], 5)
        # adj_pred = constractive_search(generated_indexes, features, adj_orig)
        gnn = GNN(adj_orig, adj_orig, features, labels, tvt_nids, print_progress=True, cuda=gpu, epochs=200, ways='js', adj_orig=adj_orig, params=params, A_pred=A_pred)
        acc, _, _ = gnn.fit()
        accs.append(acc)
    print(f'Micro F1: {np.mean(accs):.6f}, std: {np.std(accs):.6f}')
