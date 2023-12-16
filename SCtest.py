
import pickle
import numpy as np
import scipy.sparse as sp
import torch

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
        #A_probs[edges.T[0], edges.T[1]] = 0
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

def sample_graph_det(adj_orig, A_pred, remove_pct, add_pct):
    if remove_pct == 0 and add_pct == 0:
        return copy.deepcopy(adj_orig)
    orig_upper = sp.triu(adj_orig, 1)
    n_edges = orig_upper.nnz
    edges = np.asarray(orig_upper.nonzero()).T
    #把原始图中概率最低的边删去
    if remove_pct:
        n_remove = int(n_edges * remove_pct / 100)
        pos_probs = A_pred[edges.T[0], edges.T[1]]
        e_index_2b_remove = np.argpartition(pos_probs, n_remove)[:n_remove]
        mask = np.ones(len(edges), dtype=bool)
        mask[e_index_2b_remove] = False
        edges_pred = edges[mask]
    else:
        edges_pred = edges
    #将除了原始图中已有的边以外的边的概率进行排序，取前n_add个，添加进新图中
    if add_pct:
        n_add = int(n_edges * add_pct / 100)
        # deep copy to avoid modifying A_pred
        A_probs = np.array(A_pred)
        # make the probabilities of the lower half to be zero (including diagonal)
        A_probs[np.tril_indices(A_probs.shape[0])] = 0
        # make the probabilities of existing edges to be zero
        A_probs[edges.T[0], edges.T[1]] = 0
        all_probs = A_probs.reshape(-1)
        e_index_2b_add = np.argpartition(all_probs, -n_add)[-n_add:]
        new_edges = []
        for index in e_index_2b_add:
            i = int(index / A_probs.shape[0])
            j = index % A_probs.shape[0]
            new_edges.append([i, j])
        edges_pred = np.concatenate((edges_pred, new_edges), axis=0)
    adj_pred = sp.csr_matrix((np.ones(len(edges_pred)), edges_pred.T), shape=adj_orig.shape)
    adj_pred = adj_pred + adj_pred.T
    return adj_pred

def randomDropEdge(adj,droprate):
    upper = sp.triu(adj,1)
    n_edge=upper.nnz
    n_edge_left = int((1-droprate)*n_edge)
    index_edge_left = np.random.choice(n_edge, n_edge_left, replace=False)
    data = upper.data[index_edge_left]
    row = upper.row[index_edge_left]
    col = upper.col[index_edge_left]
    adj = sp.coo_matrix((data,(row,col)),shape=adj.shape)
    adj = adj+adj.T
    adj.setdiag(1)
    return adj
    
    
def calculate_laplacian_energy_distribution(X, L):
    L_csr = csr_matrix(L)
    values, U = eigsh(L_csr.toarray(), k=X.shape[0], return_eigenvectors="LM" ) #calcultate eigenvectors
    X_hat = U.T @ np.array(X) # get the post-Graph-Fourier-Transform of X
    LED = (X_hat**2) / np.sum(X_hat**2)
    #return U
    return LED , U

def check_and_fix_probabilities(p):
    # 检查是否有无效值
    if np.any(np.isnan(p)):
        # 将 NaN 替换为平均值
        avg_p = np.nanmean(p)
        p[np.isnan(p)] = avg_p
    return p

def gaussian_kernel(x,h):
    return np.exp(-x**2 / 2*h**2)

def calculate_led_shift_coefficient(LED_G, LED_S):

    SC = jensenshannon(LED_G,LED_S)
    return SC

def normalized_laplacian(adj, num_nodes):
    degree_matrix = sp.diags(np.squeeze(np.asarray(adj.sum(axis=1))), 0)
    degree_inv_sqrt = sp.diags(1.0 / np.sqrt(np.maximum(degree_matrix.diagonal(), 1e-12)), 0)
    laplacian_matrix_norm = sp.eye(num_nodes) - degree_inv_sqrt @ adj @ degree_inv_sqrt
    return laplacian_matrix_norm

def cosine_similarity(a, b):
    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    
    # 检查零向量范数，避免除以零
    if norm_a == 0 or norm_b == 0:
        return 0.0
    
    similarity = dot_product / (norm_a * norm_b) 
    return similarity

def cosine_similarity_matrix(matrix_a, matrix_b):
    # 将矩阵展平成向量
    vector_a = matrix_a.flatten()
    vector_b = matrix_b.flatten()
    similarity = cosine_similarity(vector_a, vector_b)
    
    return similarity

def euclidean_distance(matrix_a, matrix_b):
    # 将矩阵展平成向量
    vector_a = matrix_a.flatten()
    vector_b = matrix_b.flatten()

    # 计算欧几里德距离
    distance = np.linalg.norm(vector_a - vector_b)
    
    return distance


    
if __name__ == "__main__":

    adj_orig = pickle.load(open(f'data/graphs/cora_adj.pkl', 'rb'))
    features = pickle.load(open(f'data/graphs/cora_features.pkl', 'rb'))
    if sp.issparse(features):
        features = torch.FloatTensor(features.toarray())
    

    A_pred = pickle.load(open(f'data/edge_probabilities/cora_graph_2_logits.pkl', 'rb'))
    adj_pred = sample_graph_det(adj_orig, A_pred, 1, 57)
    
    num_nodes = adj_orig.shape[0]
    
    lap_norm_ori = normalized_laplacian(adj_orig, num_nodes)
    LED_ori, U_ori = calculate_laplacian_energy_distribution(features, lap_norm_ori)
    
    lap_norm_pred = normalized_laplacian(adj_pred, num_nodes)
    LED_pred, U_pred = calculate_laplacian_energy_distribution(features, lap_norm_pred)
    
    generate_indexes = generate_various_graph_1(adj_orig, A_pred, 100, 100, 5)
    for adj_generate in (generate_indexes):
        adj_generate=randomDropEdge(adj_orig, 0.4)
        lap_norm_generate = normalized_laplacian(adj_generate, num_nodes)
        LED_generate, _ = calculate_laplacian_energy_distribution(features, lap_norm_generate)
        print("cos sim", cosine_similarity_matrix(LED_ori, LED_generate))
        print("cos sim adj", cosine_similarity_matrix(csr_matrix(adj_orig),csr_matrix(adj_generate)))
    # shift = calculate_led_shift_coefficient(LED_ori.flatten(),LED_pred.flatten())
    # #sim = cosine_similarity_matrix(LED_ori,LED_pred)
    # #print(sim)
    # print("shift", shift, calculate_led_shift_coefficient(U_ori.flatten(),U_pred.flatten()))
    # print("euclidean_distance",euclidean_distance(LED_ori,LED_ori), euclidean_distance(LED_ori,LED_pred))
    # print("cos sim",cosine_similarity_matrix(LED_ori,LED_pred))
    
    #print(params["add_pct"])
    
    
   
