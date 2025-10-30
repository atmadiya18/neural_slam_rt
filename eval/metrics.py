import numpy as np

def _poses_to_centers(Ts):
    C = []
    for T in Ts:
        R = T[:3,:3]; t = T[:3,3]
        C.append((-R.T @ t).reshape(3))
    return np.asarray(C)

def _align_umeyama(P, Q):
    # Align Q -> P (Nx3) with similarity transform (s,R,t)
    Pc, Qc = P.mean(0), Q.mean(0)
    X, Y = Q - Qc, P - Pc
    S = (X.T @ Y) / len(P)
    U, Sigma, Vt = np.linalg.svd(S)
    Rm = U @ Vt
    if np.linalg.det(Rm) < 0:
        U[:,-1] *= -1
        Rm = U @ Vt
    s = np.trace(np.diag(Sigma)) / np.sum((X**2))
    t = Pc - s * (Rm @ Qc)
    return s, Rm, t

def ate_rmse(gt_T, est_T):
    """Absolute Trajectory Error (RMSE) after Umeyama alignment."""
    Pg = _poses_to_centers(gt_T)
    Pe = _poses_to_centers(est_T)
    s, Rm, t = _align_umeyama(Pg, Pe)
    Pe_aligned = (s * (Rm @ Pe.T)).T + t
    err = np.linalg.norm(Pe_aligned - Pg, axis=1)
    return float(np.sqrt(np.mean(err**2)))

def rpe_trans_rmse(gt_T, est_T, delta=1):
    """Relative Pose Error (translation RMSE) for step horizon `delta`."""
    N = min(len(gt_T), len(est_T)) - delta
    e = []
    for i in range(N):
        G_rel = np.linalg.inv(gt_T[i]) @ gt_T[i+delta]
        E_rel = np.linalg.inv(est_T[i]) @ est_T[i+delta]
        D = np.linalg.inv(G_rel) @ E_rel
        e.append(np.linalg.norm(D[:3,3]))
    e = np.asarray(e)
    return float(np.sqrt(np.mean(e**2)))
