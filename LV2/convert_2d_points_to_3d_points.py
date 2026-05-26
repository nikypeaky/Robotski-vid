import numpy as np

def convert_2d_points_to_3d_points(points_2d_L, points_2d_R, E, P):
    points_3d = np.empty((len(points_2d_L), 3))

    P = np.matrix(P)

    #svd of essential matrix
    U1, D, Vt1 = np.linalg.svd(E)
    U = U1.dot(np.diag([1,-1,-1]))
    Vt = Vt1.T.dot(np.diag([1,-1,-1])).T

    W = np.zeros((3, 3))
    W[0, 1] = -1
    W[1, 0] = 1
    W[2, 2] = 1

    A = np.matrix(U.dot(W).dot(Vt))

    b = U[:, 2]

    A_inv_b = A.I.dot(b)

    L_pi = np.empty((3, 1))
    R_pi = np.empty((3, 1))
    AR_pi = np.empty((3, 1))

    S = np.empty((2, 1))

    X = np.empty((2, 2))
    x1 = np.empty((1, 1))
    x2 = np.empty((1, 1))
    x4 = np.empty((1, 1))

    Y = np.empty((2, 1))
    y1 = np.empty((1, 1))
    y2 = np.empty((1, 1))

    #2D points in left (model) and right (scene) images
    L_m = np.empty((3, 1))
    R_m = np.empty((3, 1))

    for i in range(len(points_2d_L)):
        L_m[0, 0] = points_2d_L[i].pt[0]
        L_m[1, 0] = points_2d_L[i].pt[1]
        L_m[2, 0] = 1

        R_m[0, 0] = points_2d_R[i].pt[0]
        R_m[1, 0] = points_2d_R[i].pt[1]
        R_m[2, 0] = 1

        L_pi = P.I.dot(L_m)
        R_pi = P.I.dot(R_m)

        #init X table
        AR_pi = A.I.dot(R_pi)
        x1 = L_pi.T.dot(L_pi)
        x2 = L_pi.T.dot(AR_pi)
        x4 = AR_pi.T.dot(AR_pi)

        X[0, 0] = -x1[0, 0]
        X[0, 1] = x2[0, 0]
        X[1, 0] = x2[0, 0]
        X[1, 1] = -x4[0, 0]

        #init Y table
        y1 = L_pi.T.dot(A_inv_b)
        y2 = AR_pi.T.dot(A_inv_b)

        Y[0, 0] = -y1[0, 0]
        Y[1, 0] = y2[0, 0]

        S = np.linalg.solve(X, Y)

        s = S[0, 0]
        t = S[1, 0]

        L_pi *= s
        AR_pi *= t

        points_3d[i, :] = (L_pi + AR_pi - A_inv_b).squeeze() / 2

    return points_3d