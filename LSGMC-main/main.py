from LSGMC import LSGMC
from otherfunction.postprocessor import postprocessor
from otherfunction.retain import retain
from otherfunction.new_spectral_clustering import new_spectral_clustering
from otherfunction.AllMeasure import AllMeasure
import numpy as np
import scipy.io
import time

def load_and_preprocess_data(file_path):
    data = scipy.io.loadmat(file_path)
    X = data['X']
    Y = data['Y'].flatten()
    return X, Y

def main():


    # Load the data (assuming '3sources.mat' is in the same directory)
    data = scipy.io.loadmat('dataset/3sources.mat')

    X = data['X']  # Replace 'X' with the actual key in the .mat file
    Y = data['Y']  # Assuming Y is also in the file
    X=X[0]

    lambda_ = 100
    ppp = 0.9
    mu = 1e-2
    rho = 2.8

    # Normalize each matrix in X


    for i in range(3):
        norms = np.sqrt(np.sum(np.square(X[i]), axis=0)) + np.finfo(float).eps
        # 使用广播机制进行归一化
        X[i] = X[i] / norms
    nCluster = len(np.unique(Y))

    # Start timing
    start_time = time.time()
    Zn = LSGMC(X, lambda_, mu, rho, ppp)
    M = retain(Zn)
    W = postprocessor(M)
    label = new_spectral_clustering(W, nCluster)
    #
    # # Evaluation
    acc, nmi, F, precision, AR, _, _ = AllMeasure(label, Y)
    #
    # # Print results
    elapsed_time = time.time() - start_time
    print(f'acc {acc * 100}% nmi {nmi * 100}% time {elapsed_time}')


if __name__ == "__main__":
    main()
