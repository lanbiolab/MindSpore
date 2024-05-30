import numpy as np
from sklearn.cluster import SpectralClustering

def normr(M):
    # 按行归一化矩阵 M
    norm = np.linalg.norm(M, axis=1, keepdims=True)
    return M / norm

def new_spectral_clustering(W, numClusters):
    # 构建度矩阵 D
    D = np.diag(1.0 / np.sqrt(np.sum(W, axis=1) + np.finfo(float).eps))

    # 归一化 W
    W = D @ W @ D

    # 进行奇异值分解
    U, s, V = np.linalg.svd(W)

    # 取前 numClusters 个特征向量
    V = U[:, :numClusters]

    # 对特征向量进行归一化
    V = normr(V)

    # Spectral Clustering 聚类
    clustering = SpectralClustering(n_clusters=numClusters, n_init=200, affinity='nearest_neighbors')
    ids = clustering.fit_predict(V)

    return ids
