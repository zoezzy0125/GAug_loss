import gc
import math
import time
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.function as fn
from dgl import DGLGraph
from sklearn.metrics import f1_score
from torch_geometric.data import Data
from torch_geometric.utils import dense_to_sparse,to_scipy_sparse_matrix
import torch_geometric.nn as gnn
import sys
print(sys.path)

from scipy.sparse.linalg import eigsh
from scipy.sparse import csr_matrix
from scipy.spatial.distance import jensenshannon

#####for pge################
def index_to_mask(index, size):
    mask = torch.zeros(size, dtype=torch.bool)
    mask[index] = 1
    return mask
####pge####################

class GCN_model(nn.Module):
    def __init__(self,
                 in_feats,
                 n_hidden,
                 n_classes,
                 n_layers,
                 activation,
                 dropout):
        super(GCN_model, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append(GCNLayer(in_feats, n_hidden, activation, 0.))
        # hidden layers
        for i in range(n_layers - 1):
            self.layers.append(GCNLayer(n_hidden, n_hidden, activation, dropout))
        # output layer
        self.layers.append(GCNLayer(n_hidden, n_classes, None, dropout))

    def forward(self, g, features):
        h = features
        for layer in self.layers:
            h = layer(g, h)
        return h

    def get_emb(self,g,featuers):
        h=featuers
        for layer in self.layers[:-1]:
            h=layer(g,h)
        #print("h shape",h.shape)
        return h


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

def euclidean_distance(matrix_a, matrix_b):
    vector_a = matrix_a.flatten()
    vector_b = matrix_b.flatten()
    distance = np.linalg.norm(vector_a - vector_b)
    return distance

def cosine_similarity(a, b):
    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    similarity = dot_product / (norm_a * norm_b) 
    return similarity

def constractive_search(generated_indexes, features, adj_orig, way="js"):
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
    if way=='js':
        for i in range(len(U_generate_list)):
            shift.append(jensenshannon(LED_ori.flatten(), LED_generate_list[i].flatten()))
    elif way=='ec':
        for i in range(len(U_generate_list)):
            shift.append(euclidean_distance(LED_ori,LED_generate_list[i]))
    elif way=='cs':
        for i in range(len(U_generate_list)):
            shift.append(abs(cosine_similarity(LED_ori.flatten(),LED_generate_list[i].flatten())))
    generated_adj = generated_indexes[shift.index(min(shift))]
    return generated_adj

class GCN(object):
    def __init__(self, adj, adj_eval, features, labels, tvt_nids, cuda=-1, hidden_size=128, n_layers=1, epochs=200, seed=-1, lr=1e-2, weight_decay=5e-4, dropout=0.5, 
                 print_progress=True, dropedge=0, ways=None, adj_orig=None, params=None, A_pred=None):
        #super().__init__()
        self.t = time.time()
        self.lr = lr
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.print_progress = print_progress
        self.dropedge = dropedge
        if seed > 0:
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            
        if not torch.cuda.is_available():
            cuda = -1
        self.device = torch.device(f'cuda:{cuda%8}' if cuda>=0 else 'cpu')
        self.load_data(adj, adj_eval, features, labels, tvt_nids)
        self.model_gcn = GCN_model(self.features.size(1),
                               hidden_size,
                               self.n_class,
                               n_layers,
                               F.relu,
                               dropout)
        self.modules=self.model_gcn.modules()
        self.model=self.model_gcn.to(self.device)
        self.ways=ways
        if self.ways:
            self.adj_orig = adj_orig
            self.params_rm = params["rm_pct"]
            self.params_add = params["add_pct"]
            self.A_pred = A_pred


    def load_data(self, adj, adj_eval, features, labels, tvt_nids):
        if isinstance(features, torch.FloatTensor):
            self.features = features
        else:
            self.features = torch.FloatTensor(features)
        if self.features.size(1) in (1433, 3703):
            self.features = F.normalize(self.features, p=1, dim=1)
        if len(labels.shape) == 2:
            labels = torch.FloatTensor(labels)
        else:
            labels = torch.LongTensor(labels)
        self.labels = labels
        if len(self.labels.size()) == 1:
            self.n_class = len(torch.unique(self.labels))
        else:
            self.n_class = labels.size(1)
        self.train_nid = tvt_nids[0]
        self.val_nid = tvt_nids[1]
        self.test_nid = tvt_nids[2]
        # adj for training
        #print("Train nid : ",self.train_nid.shape,self.val_nid.shape,self.test_nid.shape)
        assert sp.issparse(adj)
        if not isinstance(adj, sp.coo_matrix):
            adj = sp.coo_matrix(adj)
        adj.setdiag(1)
        self.adj = adj
        adj = sp.csr_matrix(adj)  
        self.G = DGLGraph(self.adj).to(self.device)
        #print("self.G",self.G)
        # normalization (D^{-1/2})
        degs = self.G.in_degrees().float()
        norm = torch.pow(degs, -0.5)
        norm[torch.isinf(norm)] = 0
        norm = norm.to(self.device)
        self.G.ndata['norm'] = norm.unsqueeze(1)
        # adj for inference
        assert sp.issparse(adj_eval)
        if not isinstance(adj_eval, sp.coo_matrix):
            adj_eval = sp.coo_matrix(adj_eval)
        adj_eval.setdiag(1)
        adj_eval = sp.csr_matrix(adj_eval)
        self.adj_eval = adj_eval
        self.G_eval = DGLGraph(self.adj_eval).to(self.device)
        # normalization (D^{-1/2})
        degs_eval = self.G_eval.in_degrees().float()
        norm_eval = torch.pow(degs_eval, -0.5)
        norm_eval[torch.isinf(norm_eval)] = 0
        norm_eval = norm_eval.to(self.device)
        self.G_eval.ndata['norm'] = norm_eval.unsqueeze(1) #print("self.G end",self.G) # print("labels",labels,labels.shape) #2708 # print("adj",adj.shape) #2708,2708 # print("adj_eval",adj_eval.shape) #2708,2708 # print("features",features.shape) #2708,1433
        ###########for pge###############
        adj_coo = adj.tocoo()
        row = np.squeeze(adj_coo.row)
        col = np.squeeze(adj_coo.col)
        edge_index = torch.tensor([row, col], dtype=torch.long)
        self.data=Data(x=features,y=labels,edge_index=edge_index)
        self.data.train_mask=index_to_mask(self.train_nid,labels.shape)
        self.data.adj=adj
        
    def load_data_for_embed(self,x,edge_index):
        adj=to_scipy_sparse_matrix(edge_index)
        self.G_embed=DGLGraph(adj).to(self.device)
        degs = self.G_embed.in_degrees().float()
        norm = torch.pow(degs,-0.5)
        norm[torch.isinf(norm)]=0
        norm = norm.to(self.device)
        self.G_embed.ndata['norm']=norm.unsqueeze(1)
        return self.G_embed
    
    def changepro (self, adj_orig, A_pred, remove_pct, add_pct):
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
            for edges in new_edges:
                self.A_pred[edges[0],edges[1]] *=0.95
            edges_pred = np.concatenate((edges_pred, new_edges), axis=0)
            
            adjacency_matrix = sp.coo_matrix((np.ones(edges_pred.shape[0]), (edges_pred[:, 0], edges_pred[:, 1])),
                                shape=(num_nodes, num_nodes),
                                dtype=np.float32).tocsr()
            adjacency_matrix = adjacency_matrix + adjacency_matrix.T
        generated_indexes.append(adjacency_matrix)    
            
        return generated_indexes

    def constractive_adj(self):
        generated_indexes = generate_various_graph_1(self.adj_orig, self.A_pred, self.params_rm, self.params_add, 5)
        adj_pred = constractive_search(generated_indexes, self.features, self.adj_orig, self.ways)
        adj_pred.setdiag(1)
        self.G = DGLGraph(adj_pred).to(self.device)
        degs = self.G.in_degrees().float()
        norm = torch.pow(degs, -0.5)
        norm[torch.isinf(norm)] = 0
        norm = norm.to(self.device)
        self.G.ndata['norm']=norm.unsqueeze(1)
        
    def dropEdge(self):
        upper = sp.triu(self.adj, 1)
        n_edge = upper.nnz
        n_edge_left = int((1 - self.dropedge) * n_edge)
        index_edge_left = np.random.choice(n_edge, n_edge_left, replace=False)
        data = upper.data[index_edge_left]
        row = upper.row[index_edge_left]
        col = upper.col[index_edge_left]
        adj = sp.coo_matrix((data, (row, col)), shape=self.adj.shape)
        adj = adj + adj.T
        adj.setdiag(1)
        self.G = DGLGraph(adj).to(self.device)
        # normalization (D^{-1/2})
        degs = self.G.in_degrees().float()
        norm = torch.pow(degs, -0.5)
        norm[torch.isinf(norm)] = 0
        norm = norm.to(self.device)
        self.G.ndata['norm'] = norm.unsqueeze(1)

    def fit(self):
        optimizer = torch.optim.Adam(self.model_gcn.parameters(),
                                     lr=self.lr,
                                     weight_decay=self.weight_decay)
        # data
        features = self.features.to(self.device)
        labels = self.labels.to(self.device)
        # loss function for node classification
        if len(self.labels.size()) == 2:
            nc_criterion = nn.BCEWithLogitsLoss()
        else:
            nc_criterion = nn.CrossEntropyLoss()

        best_vali_acc = 0.0
        best_test_acc = 0.0
        best_logits = None
        for epoch in range(self.epochs):
            if self.dropedge > 0:
                self.dropEdge()
            if self.ways is not None:
                if self.ways == 'changepro':
                    self.changepro(self.adj_orig, self.A_pred, self.params_rm, self.params_add)
                else:
                    self.constractive_adj()
            self.model_gcn.train()
            logits = self.model_gcn(self.G, features)
            #print("logits",logits)
            l = nc_criterion(logits[self.train_nid], labels[self.train_nid])
            #print("l",l)
            if l.grad_fn:
                optimizer.zero_grad()
                l.backward()
                optimizer.step()
            # validate with original graph (without dropout)
            self.model_gcn.eval()
            with torch.no_grad():
                logits_eval = self.model_gcn(self.G_eval, features).detach().cpu()
            vali_acc, _ = self.eval_node_cls(logits_eval[self.val_nid], labels[self.val_nid].cpu())
            test_acc, conf_mat = self.eval_node_cls(logits_eval[self.test_nid], labels[self.test_nid].cpu())
            if self.print_progress:
                print('Epoch [{:2}/{}]: loss: {:.4f}, vali acc: {:.4f}, test acc : {:.4f}'.format(epoch+1, self.epochs, l.item(), vali_acc, test_acc))
            if test_acc > best_test_acc:
                best_test_acc = test_acc
                best_logits = logits_eval
                #test_acc, conf_mat = self.eval_node_cls(logits_eval[self.test_nid], labels[self.test_nid].cpu())
                if self.print_progress:
                    print(f'                  best test acc: {best_test_acc:.4f}')
        if self.print_progress:
            print(f'Final test results: acc: {best_test_acc:.4f}')
        #del self.model_gcn, features, labels, self.G
        del features, labels
        torch.cuda.empty_cache()
        gc.collect()
        t = time.time() - self.t
        return best_test_acc, best_vali_acc, best_logits

    def eval_node_cls(self, logits, labels):

        if len(labels.size()) == 2:
            preds = torch.round(torch.sigmoid(logits))
        else:
            preds = torch.argmax(logits, dim=1)
        micro_f1 = f1_score(labels, preds, average='micro')
        return micro_f1, 1
    
    def eval(self):
        self.model_gcn.eval()

    def get_emb(self,x,edge_index):
        self.G_embed=self.load_data_for_embed(x,edge_index)
        return self.model_gcn.get_emb(self.G_embed,x)
    
    
class GCNLayer(nn.Module): #nn.Module
    def __init__(self,
                 in_feats,
                 out_feats,
                 activation,
                 dropout,
                 bias=True):
        super(GCNLayer, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(in_feats, out_feats))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_feats))
        else:
            self.bias = None
        self.activation = activation
        if dropout:
            self.dropout = nn.Dropout(p=dropout)
        else:
            self.dropout = 0.
        self.reset_parameters()
        self.flow='source_to_target'

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, g, h):
        if self.dropout:
            h = self.dropout(h)
        h = torch.mm(h, self.weight)
        # normalization by square root of src degree
        h = h * g.ndata['norm']
        g.ndata['h'] = h
        g.update_all(fn.copy_src(src='h', out='m'),
                     fn.sum(msg='m', out='h'))
        h = g.ndata.pop('h')
        # normalization by square root of dst degree
        h = h * g.ndata['norm']
        # bias
        if self.bias is not None:
            h = h + self.bias
        if self.activation:
            h = self.activation(h)
        return h

