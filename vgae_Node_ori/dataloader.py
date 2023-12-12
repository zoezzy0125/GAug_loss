import os
import sys
import time
import pickle
import warnings
import numpy as np
import networkx as nx
import scipy.sparse as sp
import dgl
from dgl import DGLGraph
import torch
from collections import defaultdict
from sklearn.preprocessing import normalize
import torch.nn.functional as F

from utils import sparse_to_tuple

CUR_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(CUR_DIR)

class DataLoader_Node():
    def __init__(self, args):
        self.args = args
        self.dataset = args.dataset

        if self.dataset in ('cora', 'citeseer', 'pubmed'):
            self.load_data_node(self.dataset)

    def load_data_node(self, dataset):
        # load the data: x, tx, allx, graph
        tvt_nids = pickle.load(open(f'{BASE_DIR}/data/graphs/{dataset}_tvt_nids.pkl', 'rb'))
        adj = pickle.load(open(f'{BASE_DIR}/data/graphs/{dataset}_adj.pkl', 'rb'))
        #print("adj",adj,adj.shape)
        features = pickle.load(open(f'{BASE_DIR}/data/graphs/{dataset}_features.pkl', 'rb'))
        labels = pickle.load(open(f'{BASE_DIR}/data/graphs/{dataset}_labels.pkl', 'rb'))
        if sp.issparse(features):
            features = torch.FloatTensor(features.toarray())
        self.features = features
        if len(labels.shape) == 2:
            labels = torch.FloatTensor(labels)
        else:
            labels = torch.LongTensor(labels)
        self.node_label = labels
        if len(self.node_label.size()) == 1:
            self.num_class = len(torch.unique(self.node_label))
        else:
            self.num_class = self.node_label.size(1)
        
        self.train_nid = tvt_nids[0]
        self.val_nid = tvt_nids[1]
        self.test_nid = tvt_nids[2]

        if adj.diagonal().sum() > 0:
             adj = sp.coo_matrix(adj)
             adj.setdiag(0)
             adj.eliminate_zeros()
             adj = sp.csr_matrix(adj)
        
        adj_tuple = sparse_to_tuple(adj) 
        self.adj = torch.sparse.FloatTensor(torch.LongTensor(adj_tuple[0].T),
                                            torch.FloatTensor(adj_tuple[1]),
                                            torch.Size(adj_tuple[2]))
        # self.features_orig = normalize(features, norm='l1', axis=1)