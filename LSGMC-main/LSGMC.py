import numpy as np
from numpy.linalg import svd, norm
from otherfunction.spw import spw


def LSGMC(X, lambda_, mu, rho, p):
    V = len(X)
    n = X[0].shape[1]
    mu_max = 1e8
    iter_max = 30

    C = np.zeros((n, n))
    L = [np.zeros((n, n)) for _ in range(V)]
    R = [np.zeros((n, n)) for _ in range(V)]
    Z = [np.zeros((n, n)) for _ in range(V)]
    J = [np.zeros((n, n)) for _ in range(V)]
    E = [np.zeros((n, n)) for _ in range(V)]
    Q1 = [np.zeros(X[v].shape) for v in range(V)]
    Q2 = [np.zeros((n, n)) for _ in range(V)]
    Q3 = [np.zeros((n, n)) for _ in range(V)]
    tempX = [X[v].T @ X[v] for v in range(V)]

    # Optimization
    oldstop = np.zeros(V)
    for iter in range(iter_max):
        tempC = np.zeros((n, n))
        for v in range(V):
            # Update E
            tempE = Q1[v] / mu + X[v] - X[v] @ Z[v]
            for i in range(n):
                nw = np.linalg.norm(tempE[:, i])
                if nw > 1 / mu:
                    x = (nw - 1 / mu) * tempE[:, i] / nw
                else:
                    x = np.zeros(tempE[:, i].shape)
                tempE[:, i] = x
            E[v] = tempE

            # Update L, R
            tempL = (Q2[v] / mu + Z[v]) @ R[v] @ C.T
            u, _, vh = svd(tempL, full_matrices=False)
            L[v] = u @ vh
            tempR = (Q2[v] / mu + Z[v]).T @ L[v] @ C
            u, _, vh = svd(tempR, full_matrices=False)
            R[v] = u @ vh

            # Update Z
            tempZ = X[v].T @ (Q1[v] / mu - E[v]) + tempX[v] + L[v] @ C @ R[v].T - Q2[v] / mu + J[v] - Q3[v] / mu
            Z[v] = np.linalg.inv(2 * np.eye(n) + tempX[v]) @ tempZ
            tempC += L[v].T @ (Q2[v] / mu + Z[v]) @ R[v]

            # Update J
            temp = Z[v] + Q3[v] / mu
            J[v] = (temp + temp.T) / 2

        # Update C
        U, sigma, Vh = svd(tempC / V, full_matrices=False)
        xi = spw(sigma, lambda_ / (V * mu), p)  # 'spw' function needs to be defined or replaced
        C = U @ np.diag(xi) @ Vh

        # Update Q1, Q2, Q3 and mu
        for v in range(V):
            tempQ1 = X[v] - X[v] @ Z[v] - E[v]
            Q1[v] += mu * tempQ1
            tempQ2 = Z[v] - L[v] @ C @ R[v].T
            Q2[v] += mu * tempQ2
            Q3[v] += mu * (Z[v] - J[v])

        mu = min(rho * mu, mu_max)

        # Check stop condition
        stop = 0
        for v in range(V):
            tempstop = np.linalg.norm(Z[v] - C, np.inf)
            stop += (tempstop - oldstop[v])
            oldstop[v] = tempstop

        print(f'iter {iter}, stop {stop}')
        if abs(stop) < 0.01 or p == 0.5:
            break

    Zn = np.zeros((n, n))
    for v in range(V):
        Zn += Z[v]

    return Zn

