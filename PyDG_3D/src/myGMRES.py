import numpy as np
def GMRes(Af, b, x0, nmax_iter, restart=None):
    r = b - Af(x0)
    rnorm = np.linalg.norm(r)
    v = [0] * (nmax_iter)
    v[0] = r / rnorm
    h = np.zeros((nmax_iter + 1, nmax_iter))
    for k in range(0,nmax_iter):
        w = Af(v[k])
        for j in range(0,k):
            h[j, k] = np.dot(v[j], w)
            w = w - h[j, k] * v[j]
        h[k + 1, k] = np.linalg.norm(w)
        if (h[k + 1, k] != 0 and k != nmax_iter - 1):
            v[k + 1] = w / h[k + 1, k]
        b2 = np.zeros(nmax_iter + 1)
        b2[0] = rnorm
        result = np.linalg.lstsq(h, b2)[0]
        x = np.dot(np.asarray(v).transpose(), result) + x0
        r1 = b - Af(x)
        print(np.linalg.norm(r1))
    return x[:]

