import numpy as np

def retain(M):
    N = M.shape[1]
    MM = np.zeros((N, N))
    ro = 0.032 + 1 / (0.018 * N - 1.42)

    for i in range(N):
        # 对每一列的绝对值进行排序
        S = np.sort(np.abs(M[:, i]))[::-1]
        Ind = np.argsort(np.abs(M[:, i]))[::-1]

        cL1 = np.sum(S)
        stop = False
        cSum = 0
        t = 0

        while not stop:
            cSum += S[t]
            if cSum >= ro * cL1:
                stop = True
                selected_indices = Ind[:t+1]
                MM[selected_indices, i] = M[selected_indices, i]
            t += 1

    return MM
