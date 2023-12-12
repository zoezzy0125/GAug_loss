import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class VGAE(nn.Module):
    def __init__(self, adj, dim_in, dim_h, dim_z, gae): # adj, 2708, 32, 16
        super(VGAE,self).__init__()
        self.dim_z = dim_z
        self.candidate_num = 10
        self.gae = gae
        self.base_gcn = GraphConvSparse(dim_in, dim_h, adj)
        self.gcn_mean = GraphConvSparse(dim_h, dim_z, adj, activation=False)
        self.gcn_logstd = GraphConvSparse(dim_h, dim_z, adj, activation=False)
        self.gaussian_sampler = nn.Parameter(torch.randn(self.candidate_num,adj.shape[0],dim_z), requires_grad = True)

    def encode(self, X):
        hidden = self.base_gcn(X)
        self.mean = self.gcn_mean(hidden)
        if self.gae:
            # graph auto-encoder
            return self.mean
        else:
            # variational graph auto-encoder
            self.logstd = self.gcn_logstd(hidden)
            # print("mean",self.mean, self.mean.shape) #2708,16
            # gaussian_noise = torch.randn_like(self.mean) #生成与self.mean形状一样的张量，值从正态分布中抽取
            # print(self.gaussian_sampler.shape, self.logstd.shape)
            # sampled_z = gaussian_noise*torch.exp(self.logstd) + self.mean
            sampled_z = self.gaussian_sampler*torch.exp(self.logstd) + self.mean #Z = mu + xi×sigma
            #print(sampled_z.shape) #2708,16
            '''        
            # 这里使用torch.exp是因为论文中log(sigma)=GCN_{sigma}(X,A)，
            # torch.exp(self.logstd)即torch.exp(log(sigma))得到的是sigma；
            # 另外还有mu=GCN_{mu}(X,A).
            # 由于每个节点向量经过GCN后都有且仅有一个节点向量表示，
            # 所以呢，方差的对数log(sigma)和节点向量表示的均值mu分别是节点经过GCN_{sigma}(X,A)和GCN_{mu}(X,A)后得到的向量表示本身。
            # 从N(mu,sigma^2)中采样一个样本Z相当于在N(0,1)中采样一个xi，然后Z = mu + xi×sigma
            '''
            #print(self.gaussian_sampler)
            return sampled_z

    def decode(self, Z):
        Z = Z.squeeze()
        #print(Z.shape, Z.transpose(1,2).contiguous().shape)
        #print(Z.shape, Z.T.shape)
        A_pred = Z @ Z.transpose(1,2).contiguous() 
        return A_pred

    def forward(self, X):
        Z = self.encode(X)
        A_pred = self.decode(Z)
        return A_pred

class GraphConvSparse(nn.Module):
    def __init__(self, input_dim, output_dim, adj, activation=True):
        super(GraphConvSparse, self).__init__()
        self.weight = self.glorot_init(input_dim, output_dim)
        self.adj = adj
        self.activation = activation

    def glorot_init(self, input_dim, output_dim):
        init_range = np.sqrt(6.0/(input_dim + output_dim))
        initial = torch.rand(input_dim, output_dim)*2*init_range - init_range
        return nn.Parameter(initial)

    def forward(self, inputs):
        x = inputs @ self.weight
        x = self.adj @ x
        if self.activation:
            return F.elu(x)
        else:
            return x

class MyLinear(nn.Module):

    def __init__(self, hidden_dim, num_classes):
        super(MyLinear, self).__init__()
        self.linear1 = nn.Linear(hidden_dim,hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        x = self.linear1(x)
        x = self.linear2(x)
        return F.log_softmax(x, dim=1)
    