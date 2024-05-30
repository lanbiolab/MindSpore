import numpy as np
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score

def normr(M):
    norm = np.linalg.norm(M, axis=1, keepdims=True)
    return M / norm

def AllMeasure(label, true):
    # 确保 label 和 true 都是一维数组
    true = np.ravel(true)
    label = np.ravel(label)

    type = np.unique(label)
    newlabel = label.copy()
    for i in type:
        idx = np.abs(label - i) < 0.1
        truelabel = true[idx]
        typenum = np.unique(truelabel)
        nowmax = 0
        Determine = 0
        for j in typenum:
            temp2 = np.sum(np.abs(truelabel - j) < 0.1)
            if temp2 > nowmax:
                Determine = j
                nowmax = temp2
        newlabel[idx] = Determine

    # 计算指标
    acc = np.sum(np.abs(newlabel - true) < 0.1) / len(true)
    nmi = normalized_mutual_info_score(true, newlabel)

    N = len(true)
    numT = 0
    numH = 0
    numI = 0
    for n in range(N - 1):
        Tn = true[n + 1:] == true[n]
        Hn = newlabel[n + 1:] == newlabel[n]
        numT += np.sum(Tn)
        numH += np.sum(Hn)
        numI += np.sum(Tn & Hn)

    precision = numI / numH if numH > 0 else 1
    Recall = numI / numT if numT > 0 else 1
    F = 2 * precision * Recall / (precision + Recall) if (precision + Recall) > 0 else 0

    correnum = 0
    for ci in type:
        incluster = true[label == ci]
        inclunub = np.bincount(incluster)[1:]  # [1:] because bincount includes zero
        correnum += np.max(inclunub)

    Purity = correnum / N

    # 计算调整后的兰德指数
    AR = adjusted_rand_score(true, label)

    return acc, nmi, F, precision, AR, Purity, Recall
