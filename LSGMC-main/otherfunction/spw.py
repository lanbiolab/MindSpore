import numpy as np

def spw(sigma, Lambda, p):
    xi = np.zeros_like(sigma)
    if p == 1:
        for i in range(np.sum(sigma > Lambda)):
            xi[i] = sigma[i] - Lambda
    else:
        yu = (2 * Lambda * (1 - p)) ** (1 / (2 - p))
        yu += Lambda * p * yu ** (p - 1)
        idx = np.where(sigma > yu)[0]
        if len(idx) > 0:
            xi[idx] = sigma[idx]
            for j in range(3):
                xi[idx] = sigma[idx] - Lambda * p * xi[idx] ** (p - 1)
    return xi
