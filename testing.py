# essentials
import os
import csv
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
from scipy.sparse import csr_matrix, lil_matrix

# implicit framework
import implicit
from implicit.als import AlternatingLeastSquares
from implicit.approximate_als import (AnnoyAlternatingLeastSquares, FaissAlternatingLeastSquares,
                                      NMSLibAlternatingLeastSquares)
from implicit.bpr import BayesianPersonalizedRanking
from implicit.nearest_neighbours import (BM25Recommender, CosineRecommender,
                                         TFIDFRecommender, bm25_weight)
from implicit.datasets.movielens import get_movielens
from implicit.evaluation import precision_at_k, train_test_split

# utilities
import codecs
import logging
import time
import tqdm
import pickle

# evaluation.py
from evaluation import mean_roc_auc_at_k, mean_roc_auc_at_k2, mean_prec_auc_at_k

def main():
    os.environ["MKL_NUM_THREADS"] = "1"
    logging.basicConfig(level=logging.DEBUG)

    
    print(">> Main started")
    utility_matrix = pickle.load(open("pickle/utility_matrix.p", "rb"))

    coo_mat = utility_matrix.tocoo()
    train, test = train_test_split(coo_mat)
    train_user_items = train.T.tocsr()
    test_user_items = test.T.tocsr()

    model = AlternatingLeastSquares(use_gpu=True)
    model.fit(train)

    mean_roc_auc = mean_roc_auc_at_k(model, train_user_items, test_user_items, K=10)
    print(">> Mean ROC AUC score: ", mean_roc_auc)

if __name__ == '__main__':
    main()