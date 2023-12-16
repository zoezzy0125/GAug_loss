import time
import copy
import pickle
import warnings
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve, auc

def sparse_to_tuple(sparse_mx):
    if not sp.isspmatrix_coo(sparse_mx):
        sparse_mx = sparse_mx.tocoo()
    coords = np.vstack((sparse_mx.row, sparse_mx.col)).transpose()
    values = sparse_mx.data
    shape = sparse_mx.shape
    return coords, values, shape

def get_scores(edges_pos, edges_neg, A_pred, adj_label):
    preds = A_pred[edges_pos.T]
    preds_neg = A_pred[edges_neg.T]
    logists = np.hstack([preds, preds_neg])
    labels = np.hstack([np.ones(preds.size(0)), np.zeros(preds_neg.size(0))])
    # logists = A_pred.view(-1)
    # labels = adj_label.to_dense().view(-1)
    # calc scores
    roc_auc = roc_auc_score(labels, logists)
    ap_score = average_precision_score(labels, logists)
    precisions, recalls, thresholds = precision_recall_curve(labels, logists)
    pr_auc = auc(recalls, precisions)
    warnings.simplefilter('ignore', RuntimeWarning)
    f1s = np.nan_to_num(2*precisions*recalls/(precisions+recalls))
    best_comb = np.argmax(f1s)
    f1 = f1s[best_comb]
    pre = precisions[best_comb]
    rec = recalls[best_comb]
    thresh = thresholds[best_comb]
    # calc reconstracted adj_mat and accuracy with the threshold for best f1
    adj_rec = copy.deepcopy(A_pred)
    adj_rec[adj_rec < thresh] = 0
    adj_rec[adj_rec >= thresh] = 1
    labels_all = adj_label.to_dense().view(-1).long()
    preds_all = adj_rec.view(-1).long()
    recon_acc = (preds_all == labels_all).sum().float() / labels_all.size(0)
    results = {'roc': roc_auc,
               'pr': pr_auc,
               'ap': ap_score,
               'pre': pre,
               'rec': rec,
               'f1': f1,
               'acc': recon_acc,
               'adj_recon': adj_rec}
    return results

def contrastive_loss_kl(A_pred, device, margin=0.20):
    #directly calculate similarities of adjacency matrix using dor product
    #A_pred_T = A_pred_list.permute(1,2,0) #[20,2708,2708] to [2708,2708,20]
    #print(A_pred_list.shape, A_pred_T.shape)
    mask = torch.triu(torch.ones_like(A_pred)).bool()
    A_flattened = torch.masked_select(A_pred,mask).view(A_pred.size(0),-1)
    kl_divergence = torch.zeros((A_pred.size(0),A_pred.size(0)))
    for i in range(A_pred.size(0)):
        for j in range(A_pred.size(0)):
            prob_dist_i = F.softmax(A_flattened[i], dim = 0)
            prob_dist_j = F.softmax(A_flattened[j], dim = 0)
            kl_divergence[i,j] = F.kl_div(torch.log(prob_dist_i), prob_dist_j, reduction="sum")
    score_matrix = kl_divergence
    print("score_matrix",score_matrix)
    gold_score = torch.diagonal(score_matrix, offset=0)
    gold_score = torch.unsqueeze(gold_score,-1)
    difference_matrix = gold_score - score_matrix
    print("difference_matrix", difference_matrix)
    loss_matrix = margin - difference_matrix
    loss_matrix = F.relu(loss_matrix)
    #print("loss matrix", loss_matrix)
    del mask,score_matrix
    loss_mask = torch.ones_like(loss_matrix).type(torch.FloatTensor).to(device)
    diag_mask = torch.eye(loss_mask.size(1), dtype=torch.bool)
    loss_mask[diag_mask] = 0.0
    masked_loss_matrix = loss_matrix * loss_mask
    #print("masked_loss_matrix",masked_loss_matrix)
    loss_matrix = masked_loss_matrix
    loss = torch.sum(loss_matrix) / torch.sum(loss_mask)
    print("Contrastive Loss", loss)
    #loss_matrix = torch.sum(loss_matrix) / (score_matrix.shape[0]*score_matrix.shape[1])
    return loss


def contrastive_loss(A_pred, device, margin=0.20):
    #directly calculate similarities of adjacency matrix using dor product
    #A_pred_T = A_pred_list.permute(1,2,0) #[20,2708,2708] to [2708,2708,20]
    #print(A_pred_list.shape, A_pred_T.shape)
    mask = torch.triu(torch.ones_like(A_pred)).bool() #gei upper part of A_pred, to reduce space
    A_flattened = torch.masked_select(A_pred,mask).view(A_pred.size(0),-1)
    score_matrix = torch.zeros((A_pred.size(0),A_pred.size(0))).to(device)
    for i in range(A_pred.size(0)):
        for j in range(A_pred.size(0)):
            score_matrix[i,j] = F.cosine_similarity(A_flattened[i],A_flattened[j],dim=0)
    gold_score = torch.diagonal(score_matrix, offset=0)
    gold_score = torch.unsqueeze(gold_score,-1)
    difference_matrix = gold_score - score_matrix
    print("difference_matrix", difference_matrix)
    loss_matrix = margin - difference_matrix
    loss_matrix = F.relu(loss_matrix)
    #print("loss matrix", loss_matrix)
    del mask,score_matrix
    loss_mask = torch.ones_like(loss_matrix).type(torch.FloatTensor).to(device)
    diag_mask = torch.eye(loss_mask.size(1), dtype=torch.bool)
    loss_mask[diag_mask] = 0.0
    masked_loss_matrix = loss_matrix * loss_mask
    #print("masked_loss_matrix",masked_loss_matrix)
    loss_matrix = masked_loss_matrix
    loss = torch.sum(loss_matrix) / torch.sum(loss_mask)
    print("Contrastive Loss", loss)
    #loss_matrix = torch.sum(loss_matrix) / (score_matrix.shape[0]*score_matrix.shape[1])
    return loss


def train_model(args, dl, vgae):
    optimizer = torch.optim.Adam(vgae.parameters(), args.lr)
    #print("parameters", [param for param in vgae.parameters()])
    # weights for log_lik loss
    adj_t = dl.adj_train
    norm_w = adj_t.shape[0]**2 / float((adj_t.shape[0]**2 - adj_t.sum()) * 2)
    pos_weight = torch.FloatTensor([float(adj_t.shape[0]**2 - adj_t.sum()) / adj_t.sum()]).to(args.device)
    # move input data and label to gpu if needed
    features = dl.features.to(args.device)
    adj_label = dl.adj_label.to_dense().to(args.device)

    best_vali_criterion = 0.0
    best_state_dict = None
    vgae.train()
    #edgeDetermine.train()
    for epoch in range(args.epochs):
        t = time.time()
        A_pred = vgae(features)
        #A_determine = edgeDetermine(A_pred)
        #print("A_pred",A_pred,)
        #print("adj_label",adj_label) (2708*2708, 有边为1， 无边为0)
        optimizer.zero_grad()
        loss=0
        for i in range (A_pred.shape[0]):
            loss += norm_w*F.binary_cross_entropy_with_logits(A_pred[i], adj_label, pos_weight=pos_weight)
        if not args.gae:
            kl_divergence = 0.5/A_pred.size(0) * (1 + 2*vgae.logstd - vgae.mean**2 - torch.exp(2*vgae.logstd)).sum(1).mean()
            loss -= kl_divergence
        A_pred = torch.sigmoid(A_pred)
        
        loss += 10*contrastive_loss(A_pred,args.device, 0.10)
        
        A_pred = A_pred.detach().cpu()
        #roc score computation
        r_best_criterion=0
        best_r=0
        for i in range (A_pred.shape[0]):
            r = get_scores(dl.val_edges, dl.val_edges_false, A_pred[i], dl.adj_label)
            if r[args.criterion]>r_best_criterion:
                r_best_criterion = r[args.criterion]
                best_r = r
        r = best_r
        print('Epoch{:3}: train_loss: {:.4f} recon_acc: {:.4f} val_roc: {:.4f} val_ap: {:.4f} f1: {:.4f} time: {:.4f}'.format(
            epoch+1, loss.item(), r['acc'], r['roc'], r['ap'], r['f1'], time.time()-t))
        if r_best_criterion > best_vali_criterion:
            best_vali_criterion = r[args.criterion]
            best_state_dict = copy.deepcopy(vgae.state_dict())
            r_best_criterion_test = 0
            r_best_test = 0
            for i in range (A_pred.shape[0]):
                r_test = get_scores(dl.test_edges, dl.test_edges_false, A_pred[i], dl.adj_label)
                if r_test[args.criterion] > r_best_criterion_test:
                    r_best_test=r_test
            r_test = r_best_test
            print("          test_roc: {:.4f} test_ap: {:.4f} test_f1: {:.4f} test_recon_acc: {:.4f}".format(
                    r_test['roc'], r_test['ap'], r_test['f1'], r_test['acc']))
        loss.backward()
        optimizer.step()

    print("Done! final results: test_roc: {:.4f} test_ap: {:.4f} test_f1: {:.4f} test_recon_acc: {:.4f}".format(
            r_test['roc'], r_test['ap'], r_test['f1'], r_test['acc']))

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
        print(adj_recon)
        np.fill_diagonal(adj_recon, 0)
        # np.ndarray
        if args.gae:
            filename = f'graphs/{args.dataset}_graph_{i+1}_logits_gae.pkl'
        else:
            filename = f'graphs/{args.dataset}_graph_{i+1}_logits.pkl'
        pickle.dump(adj_recon, open(filename, 'wb'))
