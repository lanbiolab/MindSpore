import numpy as np

def compute_nmi(T, H):
    N = len(T)
    classes = np.unique(T)
    clusters = np.unique(H)
    num_class = len(classes)
    num_clust = len(clusters)

    # Compute number of points in each class
    D = np.array([np.sum(T == c) for c in classes])

    # Mutual information
    mi = 0
    A = np.zeros((num_clust, num_class))
    avgent = 0
    for i in range(num_clust):
        index_clust = H == clusters[i]
        B_i = np.sum(index_clust)
        for j in range(num_class):
            index_class = T == classes[j]
            A[i, j] = np.sum(index_class & index_clust)
            if A[i, j] != 0:
                mi_arr_ij = A[i, j] / N * np.log2(N * A[i, j] / (B_i * D[j]))
                avgent -= (B_i / N) * (A[i, j] / B_i) * np.log2(A[i, j] / B_i)
            else:
                mi_arr_ij = 0
            mi += mi_arr_ij

    # Class entropy
    class_ent = np.sum(D / N * np.log2(N / D))

    # Clustering entropy
    clust_ent = np.sum(np.array([np.sum(H == c) for c in clusters]) / N * np.log2(N / np.array([np.sum(H == c) for c in clusters])))

    # Normalized mutual information
    nmi = 2 * mi / (clust_ent + class_ent)

    return A, nmi, avgent
