'''
This .py file contains MCMC sampling codes
'''

import os
import sys
import torch
import pickle
import argparse
from torch import Tensor
from torch_geometric.datasets import Planetoid, PPI, Flickr, Airports
from torch_geometric.utils import degree
from torch_geometric.typing import Adj
import torch_geometric.transforms as T
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
import numpy as np
import math
from scipy.stats import norm
from sklearn.mixture import GaussianMixture 
import matplotlib.pyplot as plt
sys.path.append(os.path.join(os.path.dirname("__file__"), '..'))
import random
###############For explainer

from torch_geometric.explain.algorithm import PGExplainer
from torch.nn import ReLU, Sequential

from torch_geometric.explain import Explanation,Explainer
from torch_geometric.explain.algorithm import ExplainerAlgorithm
from torch_geometric.explain.algorithm.utils import clear_masks, set_masks
from torch_geometric.explain.config import (
    ModelConfig,
)

import scipy.sparse as sp
from scipy.sparse import csr_matrix, coo_matrix
##############

# parse parameters
parser = argparse.ArgumentParser()
parser.add_argument('-d', '--dataset', type=str, default='Cora',
                    help='The dataset to be trained on [Cora, CiteSeer, PubMed].')
parser.add_argument('-c', '--cuda-device', type=int, default=0, help='which gpu device to use.')
parser.add_argument('-dr', '--dropping-rate', type=float, default=0, help='The chosen dropping rate (default: 0).')
parser.add_argument('-e', '--epoch', type=int, default=500, help='The epoch number (default: 500).')
parser.add_argument('-ee', '--epoch_exp', type=int, default=30, help='The epoch number of explainer (default: 30).')
parser.add_argument('-seed','--seed', type=int, default=7)
args = parser.parse_args()

if args.seed > 0:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        print("seed",args.seed)
# device selection
device = torch.device('cuda:{}'.format(args.cuda_device) if torch.cuda.is_available() else 'cpu')


from torch_geometric.nn import GCNConv
from torch_geometric.utils import dropout_edge
# Model
class Model_DropEdge(torch.nn.Module):
    def __init__(self,feature_num,output_num,hidden_layer_dimension):
        super(Model_DropEdge,self).__init__()
        self.gnn1=GCNConv(feature_num,hidden_layer_dimension)
        self.gnn2=GCNConv(hidden_layer_dimension,hidden_layer_dimension)
        self.gnn3=GCNConv(hidden_layer_dimension,output_num)
    
    def forward(self,x:Tensor, edge_index:Adj,drop_rate: float=0):
        #print(edge_index)
        edge_index_retain,edge_mask=dropout_edge(edge_index,p=drop_rate)
        x=self.gnn1(x,edge_index_retain)
        x=F.relu(x)
        x=self.gnn2(x,edge_index_retain)
        x=F.relu(x)
        x=self.gnn3(x,edge_index_retain)
        return x
    
    def reset_parameters(self):
        self.gnn1.reset_parameters()
        self.gnn2.reset_parameters()
        self.gnn3.reset_parameters()

tvt_nids = pickle.load(open(f'data/graphs/{args.dataset}_tvt_nids.pkl', 'rb'))
adj_orig = pickle.load(open(f'data/graphs/{args.dataset}_adj.pkl', 'rb'))
features = pickle.load(open(f'data/graphs/{args.dataset}_features.pkl', 'rb'))
labels = pickle.load(open(f'data/graphs/{args.dataset}_labels.pkl', 'rb'))
if sp.issparse(features):
    features = torch.FloatTensor(features.toarray())

print("features",features, features.shape) #for ppi, 10076,50
num_features = features.size(1)
print("num_features", num_features)
num_classes = len(torch.unique(torch.tensor(labels)))
print("num_classes", num_classes)
coo_mat = coo_matrix(adj_orig)
row_indices = coo_mat.row.tolist()
col_indices = coo_mat.col.tolist()
edge_index = torch.tensor([row_indices, col_indices])
print("edge_index", edge_index.shape)
num_nodes = labels.shape[0]
print("num_nodes",num_nodes)

features = features.to(device)
print("features", features)
labels = torch.tensor(labels).to(device)
edge_index = edge_index.to(device)

num_x = features.size(0)
train_mask = torch.isin(torch.arange(num_x), torch.tensor(tvt_nids[0])).to(device)
val_mask = torch.isin(torch.arange(num_x), torch.tensor(tvt_nids[1])).to(device)
test_mask = torch.isin(torch.arange(num_x), torch.tensor(tvt_nids[2])).to(device)

hidden_layer_dimension=128
model = Model_DropEdge(num_features, num_classes,hidden_layer_dimension).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=0.0005)
epoch_num = args.epoch
epoch_exp=args.epoch_exp

def train(model):
    model.train()
    optimizer.zero_grad()
    out = model(features, edge_index, args.dropping_rate)
    loss = F.cross_entropy(out[train_mask], labels[train_mask])
    loss.backward()
    #print('the train loss is {}'.format(float(loss)))
    optimizer.step()


@torch.no_grad()
def test(model):
    model.eval()
    out = model(features, edge_index)
    _, pred = out.max(dim=1)
    train_correct = int(pred[train_mask].eq(labels[train_mask]).sum().item())
    train_acc = train_correct / int(train_mask.sum())
    validate_correct = int(pred[val_mask].eq(labels[val_mask]).sum().item())
    validate_acc = validate_correct / int(val_mask.sum())
    test_correct = int(pred[test_mask].eq(labels[test_mask]).sum().item())
    test_acc = test_correct / int(test_mask.sum())
    return train_acc, validate_acc, test_acc

#model.load_state_dict(torch.load('result/GCNTest.pkl'))
best_test_acc = test_acc = 0
for epoch in range(epoch_num):
    train(model)
    train_acc, val_acc, current_test_acc = test(model)
    print('For the {} epoch, the train acc is {}, the val acc is {}, the test acc is {}.'.format(epoch, train_acc,
                                                                                                 val_acc,
                                                                                                 current_test_acc))
    if current_test_acc > best_test_acc:
        best_test_acc = current_test_acc
        torch.save(model.state_dict(),'GCN'+str(args.dataset)+'Test.pkl')
        
print("Final model holds test acc:",best_test_acc)
model.load_state_dict(torch.load('GCN'+str(args.dataset)+'Test.pkl'))
explainer = Explainer(
            model=model,
            algorithm=PGExplainer(epochs=epoch_exp, lr=0.003).to(device),
            explanation_type='phenomenon',
            edge_mask_type='object',
            model_config=ModelConfig(mode='multiclass_classification',task_level='node',return_type='raw')
        )


epoch_num = args.epoch

exp_indices=torch.where(train_mask==True)[0]
#print(data)
#print(len(exp_indices))
for epoch in range(epoch_exp):
    explainer_epoch_loss=0
    for index in exp_indices:
        #print("features",features)
        loss=explainer.algorithm.train(epoch,model,features,edge_index,target=labels,index=index)
        explainer_epoch_loss+=loss   
        #explanation=explainer()
    print("Explainer epoch {} loss is {} ".format(epoch,explainer_epoch_loss))

edge_mask_list = []
deg = degree(edge_index[0], dtype = torch.float)
deg_weight = 1/deg[train_mask] 
for node_id in torch.where(train_mask)[0]:
    #print("features", features)
    explanation_before=explainer(features, edge_index, index=node_id, target = labels)
    edge_mask_list.append(explanation_before.edge_mask)
edge_mask_mean = deg_weight@(torch.stack(edge_mask_list))
edge_deg_weight = torch.zeros(num_nodes, num_nodes)
for i in range (len(edge_mask_mean)):
    edge_deg_weight[edge_index[0][i],edge_index[1][i]]=edge_mask_mean[i]
    
with open('deg_weight/'+args.dataset+".pkl", 'wb') as f:
    pickle.dump(edge_deg_weight.cpu(), f)



print('Mission completes, weight file save to /{args.dataset}.pkl')


# with open ("result/GCNTestpge.pkl",'wb') as f:
#     pickle.dump(explainer,f)