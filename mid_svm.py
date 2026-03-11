"""
Simplified implementation of Multiple Incremental Decremental SVM (MID-SVM)
Based on: Karasuyama & Takeuchi, "Multiple Incremental Decremental Learning of Support Vector Machines", NeurIPS 2009

This module implements a simplified path-following algorithm for adding multiple points to an SVM.
"""

import numpy as np
from sklearn.svm import SVC
from sklearn.metrics.pairwise import rbf_kernel, linear_kernel
import time


def _rbf_kernel(X1, X2, gamma):
    """RBF kernel K(x,y) = exp(-gamma * ||x-y||^2)"""
    return rbf_kernel(X1, X2, gamma=gamma)


def _linear_kernel(X1, X2):
    """Linear kernel K(x,y) = x^T y"""
    return np.dot(X1, X2.T)


def get_kernel_matrix(X1, X2, y1, y2, kernel='rbf', gamma=0.5):
    """Compute Q_ij = y_i * y_j * K(x_i, x_j)"""
    if kernel == 'rbf':
        K = _rbf_kernel(X1, X2, gamma)
    else:
        K = _linear_kernel(X1, X2)
    y1 = np.asarray(y1).reshape(-1, 1)
    y2 = np.asarray(y2).reshape(1, -1)
    return np.outer(y1, y2) * K


def incremental_svm_add_multiple(X_train, y_train, X_add, y_add, C=1.0, gamma=0.5, 
                                   kernel='rbf', random_state=42):
    """
    Simplified MID-SVM: Add multiple points to an existing SVM using path-following.
    
    Uses Equations (3), (4), (7), (10), (11) from the paper.
    Returns: (alpha_full, b, n_breakpoints, elapsed_time)
    """
    np.random.seed(random_state)
    n_orig = len(X_train)
    n_add = len(X_add)
    
    X_all = np.vstack([X_train, X_add])
    y_all = np.concatenate([y_train, y_add])
    
    # Train initial SVM on original data (Section 2)
    svc = SVC(C=C, kernel=kernel, gamma=gamma if kernel == 'rbf' else 'scale', 
              tol=1e-6, random_state=random_state)
    svc.fit(X_train, y_train)
    
    # Extract solution: sklearn dual_coef_[0][i] = alpha_i * y_i for support vector i
    # Alpha must be non-negative; we use alpha_i = |dual_coef_[0][i]|
    alpha = np.zeros(n_orig + n_add)
    sv_indices = svc.support_
    alpha_sv = np.abs(svc.dual_coef_).flatten()
    for i, idx in enumerate(sv_indices):
        if idx < n_orig:
            alpha[idx] = alpha_sv[i]
    
    # Bias
    b = svc.intercept_[0]
    
    # Compute Q matrix (Eq. 2, Q_ij = y_i y_j K(x_i, x_j))
    Q = get_kernel_matrix(X_all, X_all, y_all, y_all, kernel, gamma)
    
    # Add small constant to diagonal for numerical stability (paper Section 4)
    Q += 1e-6 * np.eye(Q.shape[0])
    
    # Index sets (Eq. 2a-2c): O, M, I
    f = np.array([np.sum(alpha * y_all * Q[i, :]) + b for i in range(n_orig)])
    yf = y_all[:n_orig] * (np.sum(alpha[:n_orig] * y_all[:n_orig] * Q[:n_orig, :n_orig], axis=1) + b)
    
    # Recompute f for all points using current alpha
    def compute_f(alpha_vec, b_val):
        return np.array([np.sum(alpha_vec * y_all * Q[i, :]) + b_val for i in range(len(alpha_vec))])
    
    O = set()
    M = set()
    I = set()
    for i in range(n_orig):
        yi_fi = y_all[i] * (np.sum(alpha * y_all * Q[i, :]) + b)
        if yi_fi > 1 + 1e-6:
            O.add(i)
        elif yi_fi < 1 - 1e-6:
            I.add(i)
        else:
            M.add(i)
    
    A = set(range(n_orig, n_orig + n_add))
    R = set()
    
    # Remove from A points that already satisfy optimality (Section 3.2)
    to_remove = []
    for i in A:
        yi_fi = y_all[i] * (np.sum(alpha * y_all * Q[i, :]) + b)
        if yi_fi > 1 + 1e-6:
            O.add(i)
            to_remove.append(i)
        elif abs(yi_fi - 1) < 1e-6:
            M.add(i)
            to_remove.append(i)
    for i in to_remove:
        A.discard(i)
    
    n_breakpoints = 0
    start_time = time.perf_counter()
    max_iter = 5000
    
    for _ in range(max_iter):
        if len(A) == 0 and len(R) == 0:
            break
            
        M_list = sorted(M)
        A_list = sorted(A)
        R_list = sorted(R)
        
        if len(M_list) == 0:
            break
        
        # Build M matrix (Eq. 7)
        y_M = y_all[M_list]
        Q_M = Q[np.ix_(M_list, M_list)]
        M_mat = np.block([[0, y_M.reshape(1, -1)], [y_M.reshape(-1, 1), Q_M]])
        
        # Right-hand side: [y_A^T y_R^T; Q_M,A Q_M,R] * [C*1 - alpha_A; -alpha_R]
        if len(A_list) > 0:
            dA = C - np.array([alpha[i] for i in A_list])
            rhs_A = np.concatenate([[np.sum(y_all[A_list] * dA)], 
                                     np.sum(Q[np.ix_(M_list, A_list)] * dA, axis=1)])
        else:
            rhs_A = np.zeros(len(M_list) + 1)
            
        if len(R_list) > 0:
            aR = np.array([alpha[i] for i in R_list])
            rhs_R = np.concatenate([[np.sum(y_all[R_list] * aR)], 
                                    np.sum(Q[np.ix_(M_list, R_list)] * aR, axis=1)])
        else:
            rhs_R = np.zeros(len(M_list) + 1)
        
        rhs = rhs_A - rhs_R
        
        try:
            phi = -np.linalg.solve(M_mat, rhs)
        except np.linalg.LinAlgError:
            break
        
        # psi for step length (Eq. 11): y * Delta_f = eta * psi
        Delta_b = phi[0]
        Delta_alpha_M = phi[1:]
        
        psi = np.zeros(n_orig + n_add)
        for i in range(n_orig + n_add):
            delta_f = Delta_b
            for j_idx, j in enumerate(M_list):
                delta_f += Delta_alpha_M[j_idx] * Q[i, j]
            for j in A_list:
                delta_f += (C - alpha[j]) * Q[i, j]
            for j in R_list:
                delta_f -= alpha[j] * Q[i, j]
            psi[i] = y_all[i] * delta_f
        
        # Compute step length eta from constraints (8), (9)
        eta_candidates = [1.0]
        
        for i in M_list:
            if abs(Delta_alpha_M[M_list.index(i)]) > 1e-12:
                eta_lo = -alpha[i] / Delta_alpha_M[M_list.index(i)]
                eta_hi = (C - alpha[i]) / Delta_alpha_M[M_list.index(i)]
                if eta_lo > 0:
                    eta_candidates.append(eta_lo)
                if eta_hi > 0:
                    eta_candidates.append(eta_hi)
        
        for i in O:
            if psi[i] < -1e-12:
                eta_candidates.append((1 - y_all[i] * (np.sum(alpha * y_all * Q[i, :]) + b)) / psi[i])
        for i in I:
            if psi[i] > 1e-12:
                eta_candidates.append((1 - y_all[i] * (np.sum(alpha * y_all * Q[i, :]) + b)) / psi[i])
        for i in A_list:
            if psi[i] > 1e-12:
                eta_candidates.append((1 - y_all[i] * (np.sum(alpha * y_all * Q[i, :]) + b)) / psi[i])
        
        eta_candidates = [e for e in eta_candidates if e > 1e-10 and e <= 1.0]
        eta = min(eta_candidates) if eta_candidates else 1.0
        
        # Update parameters (Eq. 3, 4, 10)
        for j_idx, j in enumerate(M_list):
            alpha[j] += eta * Delta_alpha_M[j_idx]
        b += eta * Delta_b
        for j in A_list:
            alpha[j] += eta * (C - alpha[j])
        for j in R_list:
            alpha[j] -= eta * alpha[j]
        
        n_breakpoints += 1
        
        # Update index sets
        for i in list(M_list):
            yi_fi = y_all[i] * (np.sum(alpha * y_all * Q[i, :]) + b)
            if alpha[i] <= 1e-10:
                M.discard(i)
                O.add(i)
            elif alpha[i] >= C - 1e-10:
                M.discard(i)
                I.add(i)
            elif yi_fi > 1 + 1e-6:
                M.discard(i)
                O.add(i)
            elif yi_fi < 1 - 1e-6:
                M.discard(i)
                I.add(i)
        
        for i in list(A_list):
            yi_fi = y_all[i] * (np.sum(alpha * y_all * Q[i, :]) + b)
            if abs(yi_fi - 1) < 1e-6:
                A.discard(i)
                M.add(i)
            elif alpha[i] >= C - 1e-10:
                A.discard(i)
                I.add(i)
        
        for i in list(R_list):
            if alpha[i] <= 1e-10:
                R.discard(i)
        
        if eta >= 1 - 1e-6:
            break
    
    elapsed = time.perf_counter() - start_time
    return alpha, b, n_breakpoints, elapsed


def single_incremental_add(X_train, y_train, X_add, y_add, C=1.0, gamma=0.5, 
                           kernel='rbf', random_state=42):
    """
    Simulate SID-SVM: Add points one at a time (repeated single-point updates).
    Uses batch retraining for each addition as a proxy (simpler than full SID-SVM).
    """
    X_cur = X_train.copy()
    y_cur = y_train.copy()
    total_time = 0
    
    for i in range(len(X_add)):
        X_cur = np.vstack([X_cur, X_add[i:i+1]])
        y_cur = np.concatenate([y_cur, [y_add[i]]])
        start = time.perf_counter()
        svc = SVC(C=C, kernel=kernel, gamma=gamma if kernel == 'rbf' else 'scale', 
                  tol=1e-6, random_state=random_state)
        svc.fit(X_cur, y_cur)
        total_time += time.perf_counter() - start
    
    return total_time


def batch_retrain(X_train, y_train, X_add, y_add, C=1.0, gamma=0.5, 
                  kernel='rbf', random_state=42):
    """Batch retraining: train SVM on all data from scratch."""
    X_all = np.vstack([X_train, X_add])
    y_all = np.concatenate([y_train, y_add])
    start = time.perf_counter()
    svc = SVC(C=C, kernel=kernel, gamma=gamma if kernel == 'rbf' else 'scale', 
              tol=1e-6, random_state=random_state)
    svc.fit(X_all, y_all)
    return time.perf_counter() - start
