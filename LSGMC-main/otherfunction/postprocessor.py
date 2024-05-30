import numpy as np
def normr(M):
    # 按行归一化矩阵 M
    norm = np.linalg.norm(M, axis=1, keepdims=True)
    return M / norm
def postprocessor(Zn):
    # 进行奇异值分解
    U, s, Vh = np.linalg.svd(Zn)

    # 提取奇异值并确定有效的 r 值
    r = np.sum(s > 1e-6)

    # 提取前 r 个奇异向量和奇异值
    U_r = U[:, :r]
    s_r = np.diag(s[:r])

    # 计算 M
    M = U_r @ np.sqrt(s_r)

    # 归一化 M 并计算相关性矩阵
    mm = normr(M)
    rs = mm @ mm.T

    # 计算 W
    W = np.power(rs, 2)

    return W