import logging
import time
import tqdm

import numpy as np
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc

def mean_roc_auc_at_k(model, train_user_items, test_user_items, K=10, show_progress=True):
    auc_list = []
    lenders_count, loans_count = train_user_items.shape
    start = time.time()
    
    with tqdm.tqdm(total=lenders_count) as progress:
        for lender_index in range(lenders_count):
            if test_user_items[lender_index, :].nnz == 0:
                continue
                
            lender_row = np.zeros(loans_count)
            for loan_index, score in model.recommend(lender_index, train_user_items, N=K):
                lender_row[loan_index] = score
            
            test_lender_row = test_user_items[lender_index, :].toarray().flatten()
            roc_auc = roc_auc_score(test_lender_row, lender_row)
            auc_list.append(roc_auc)
            progress.update(1)
            
    logging.debug("generated mean ROC AUC in %0.2fs", time.time() - start)
    return np.mean(auc_list)  

def mean_prec_auc_at_k(model, train_user_items, test_user_items, K=10, show_progress=True):
    auc_list = []
    lenders_count, loans_count = train_user_items.shape
    start = time.time()
    
    with tqdm.tqdm(total=lenders_count) as progress:
        for lender_index in range(lenders_count):
            if test_user_items[lender_index, :].nnz == 0:
                continue
                
            lender_row = np.zeros(loans_count)
            for loan_index, score in model.recommend(lender_index, train_user_items, N=K):
                lender_row[loan_index] = score
            
            test_lender_row = test_user_items[lender_index, :].toarray().flatten()
            
            precision, recall, thresholds = precision_recall_curve(test_lender_row, lender_row, pos_label=1)
            prec_auc = auc(recall, precision)                
            auc_list.append(prec_auc)
            progress.update(1)
            
    logging.debug("generated mean Precision/Recall curve AUC in %0.2fs", time.time() - start)
    return np.mean(auc_list)

def mean_roc_auc_at_k2(model, train_user_items, test_user_items, K=10, show_progress=True):
    auc_list = []
    lenders_count, loans_count = train_user_items.shape
    start = time.time()
    
    with tqdm.tqdm(total=lenders_count) as progress:
        for lender_index in range(lenders_count):
            if test_user_items[lender_index, :].nnz == 0:
                continue
                
            lender_vect = model.user_factors[lender_index]
            liked = train_user_items[lender_index].indices

            scores = model.item_factors.dot(lender_vect)
            scores = np.delete(scores, liked)
            
            test_lender_row = test_user_items[lender_index, :].toarray().flatten()
            test_lender_row = np.delete(test_lender_row, liked)
            
            roc_auc = roc_auc_score(test_lender_row, scores)
            auc_list.append(roc_auc)
            progress.update(1)
            
    logging.debug("generated mean ROC AUC in %0.2fs", time.time() - start)
    return np.mean(auc_list)  