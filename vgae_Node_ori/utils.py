import time
import copy
import pickle
import warnings
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve, auc, roc_curve, f1_score
import matplotlib.pyplot as plt
import torch.nn as nn

def sparse_to_tuple(sparse_mx):
    if not sp.isspmatrix_coo(sparse_mx):
        sparse_mx = sparse_mx.tocoo()
    coords = np.vstack((sparse_mx.row, sparse_mx.col)).transpose()
    values = sparse_mx.data
    shape = sparse_mx.shape
    return coords, values, shape

def draw_curve(edges_pos, edges_neg, A_pred, adj_label):
    preds = A_pred[edges_pos.T]
    preds_neg = A_pred[edges_neg.T]
    logists = np.hstack([preds, preds_neg])
    labels = np.hstack([np.ones(preds.size(0)), np.zeros(preds_neg.size(0))])
    precisions, recalls, thresholds = precision_recall_curve(labels, logists)
    pr_auc = auc(recalls, precisions)
    ###draw pr_curve
    plt.figure(figsize=(10,6))
    plt.figure(figsize=(10, 6))
    plt.plot(recalls, precisions, color='darkorange', lw=2, label='PR curve (area = {:.2f})'.format(pr_auc))
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc='upper right')
    plt.show()
    plt.savefig("pr_curve.png")
    
    fpr, tpr, thresholds = roc_curve(labels, logists)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(10, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = {:.2f})'.format(roc_auc))
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc='lower right')
    plt.savefig("roc_curve.png")
    

def train_model(args, dl, vgae):
    '''
    #Todo:
    1. label: Label for node,  #?1 
    2. dl.train_nid #?2
    3. dl.num_class #?3
    '''

    optimizer = torch.optim.Adam(vgae.parameters(), lr=args.lr)
    features = dl.features.to(args.device)
    label = dl.node_label.to(args.device)
    

    if dl.num_class == 2: #?3
        nc_criterion = nn.BCEWithLogitsLoss()
    else:
        nc_criterion = nn.CrossEntropyLoss()
    
    best_vali_acc = 0.0
    best_state_dict = None
    for epoch in range(args.epochs):
        vgae.train()
        logits = vgae(features)
        loss = nc_criterion(logits[dl.train_nid], label[dl.train_nid]) #?1 #?2
        kl_divergence = 0.5/features.shape[0] * (1 + 2*vgae.logstd - vgae.mean**2 - torch.exp(2*vgae.logstd)).sum(1).mean()
        loss -=kl_divergence
        if loss.grad_fn:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        vgae.eval()
        logits = logits.detach().cpu()
        logits = torch.argmax(logits, dim=1)
        #print(logits, logits.shape)
        vali_acc = f1_score(logits[dl.val_nid], label[dl.val_nid].cpu(), average='micro')
        
        print('Epoch [{:2}/{}]: loss: {:.4f}, vali acc: {:.4f}'.format(epoch+1, args.epochs, loss.item(), vali_acc))
        if vali_acc > best_vali_acc:
            best_vali_acc = vali_acc
            best_state_dict = copy.deepcopy(vgae.state_dict())
            test_acc = f1_score(logits[dl.test_nid], label[dl.test_nid].cpu(), average='micro')
            print('Test acc : {:.4f} '.format(test_acc))
    print("Done! final results: test acc : {:.4f}".format(
            test_acc))

    vgae.load_state_dict(best_state_dict)
    return vgae

def gen_graphs(args, dl, vgae):
    adj_orig = dl.adj_orig
    assert adj_orig.diagonal().sum() == 0
    # sp.csr_matrix
    if args.gae:
        pickle.dump(adj_orig, open(f'graphs/{args.dataset}_graph_0_gae.pkl', 'wb'))
    else:
        pickle.dump(adj_orig, open(f'graphs/{args.dataset}_graph_0.pkl', 'wb'))
    # sp.lil_matrix
    pickle.dump(dl.features_orig, open(f'graphs/{args.dataset}_features.pkl', 'wb'))
    features = dl.features.to(args.device)
    for i in range(args.gen_graphs):
        with torch.no_grad():
            A_pred = vgae(features)
        A_pred = torch.sigmoid(A_pred).detach().cpu()
        r = get_scores(dl.val_edges, dl.val_edges_false, A_pred, dl.adj_label)
        adj_recon = A_pred.numpy()
        np.fill_diagonal(adj_recon, 0)
        # np.ndarray
        if args.gae:
            filename = f'graphs/{args.dataset}_graph_{i+1}_logits_gae.pkl'
        else:
            filename = f'graphs/{args.dataset}_graph_{i+1}_logits.pkl'
        pickle.dump(adj_recon, open(filename, 'wb'))