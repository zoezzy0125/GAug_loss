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
    
def get_scores(edges_pos, edges_neg, A_pred, adj_label):
    # get logists and labels
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
    #print("thresh",thresh)
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

def train_model(args, dl, vgae, linear_model=None):
    optimizer = torch.optim.Adam(vgae.parameters(), lr=args.lr)
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
    for epoch in range(args.epochs):
        t = time.time()
        A_pred = vgae(features)
        optimizer.zero_grad()
        loss = log_lik = norm_w*F.binary_cross_entropy_with_logits(A_pred, adj_label, pos_weight=pos_weight)
        if not args.gae:
            kl_divergence = 0.5/A_pred.size(0) * (1 + 2*vgae.logstd - vgae.mean**2 - torch.exp(2*vgae.logstd)).sum(1).mean()
            loss -= kl_divergence

        A_pred = torch.sigmoid(A_pred).detach().cpu()
        r = get_scores(dl.val_edges, dl.val_edges_false, A_pred, dl.adj_label)
        print('Epoch{:3}: train_loss: {:.4f} recon_acc: {:.4f} val_roc: {:.4f} val_ap: {:.4f} f1: {:.4f} time: {:.4f}'.format(
            epoch+1, loss.item(), r['acc'], r['roc'], r['ap'], r['f1'], time.time()-t))
        if r[args.criterion] > best_vali_criterion:
            best_vali_criterion = r[args.criterion]
            best_state_dict = copy.deepcopy(vgae.state_dict())
            r_test = get_scores(dl.test_edges, dl.test_edges_false, A_pred, dl.adj_label)
            draw_curve(dl.test_edges, dl.test_edges_false, A_pred, dl.adj_label)
            print("          test_roc: {:.4f} test_ap: {:.4f} test_f1: {:.4f} test_recon_acc: {:.4f}".format(
                    r_test['roc'], r_test['ap'], r_test['f1'], r_test['acc']))
        loss.backward()
        optimizer.step()

    print("Done! final results: test_roc: {:.4f} test_ap: {:.4f} test_f1: {:.4f} test_recon_acc: {:.4f}".format(
            r_test['roc'], r_test['ap'], r_test['f1'], r_test['acc']))

    vgae.load_state_dict(best_state_dict)
    if linear_model is not None:
        vgae.eval()
        embeddings = vgae.encode(features).detach()
        model = linear_model
        optimizer = torch.optim.Adam(model.parameters(),lr = args.lr)
        
        best_vali_acc = 0.0
        best_state_dict = None
        for epoch in range (args.nc_epochs):
            model.train()
            optimizer.zero_grad()
            output = model(embeddings)
            loss = F.cross_entropy(output[dl.train_nid], dl.labels[dl.train_nid].to(args.device))
            loss.backward()
            optimizer.step()
            
            output = output.detach().cpu()
            output = torch.argmax(output, dim=1)
            vali_acc = f1_score(output[dl.val_nid], dl.labels[dl.val_nid].cpu(), average = 'micro')
            print('Epoch [{:2}/{}]: loss: {:.4f}, vali acc: {:.4f}'.format(epoch+1, args.nc_epochs, loss.item(), vali_acc))
            if vali_acc > best_vali_acc:
                best_vali_acc = vali_acc
                best_state_dict = copy.deepcopy(linear_model.state_dict())
                test_acc = f1_score(output[dl.test_nid], dl.labels[dl.test_nid].cpu(), average='micro')
                print('Test acc : {:.4f} '.format(test_acc))
    print("Done! final results: test acc : {:.4f}".format(
            test_acc))
    linear_model.load(best_state_dict)
    return vgae, linear_model

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