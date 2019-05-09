import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix, lil_matrix
import pickle

def main():
    utility_matrix = pickle.load(open("pickle/utility_matrix.p", "rb"))
    utility_matrix = utility_matrix.astype('int8')
    sdf = pd.SparseDataFrame(utility_matrix)
    print(sdf.memory_usage())

if __name__ == '__main__':
    main()