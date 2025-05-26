import numpy as np
import scipy.sparse as sp
from datetime import datetime
from time import process_time


def solve_linear_system(method, A, b, x_exact, tol=1e-6, max_iter=20000):
    """
    Dispatcher che richiama il metodo iterativo specificato
    Args:
        method (str): Nome del metodo ['jacobi', 'gauss_seidel', 'gradient', 'conjugate_gradient']
        A: Matrice del sistema
        b: Vettore dei termini noti
        x_exact: Soluzione esatta
        tol: Tolleranza
        max_iter: Numero massimo di iterazioni

    Returns:
        tuple: (soluzione, iterazioni, tempo, errore_relativo)
    """
    method = method.lower().replace('-', '_').replace(' ', '_')

    method_map = {
        'jacobi': jacobi,
        'gauss_seidel': gauss_seidel,
        'gs': gauss_seidel,
        'gradient': gradient,
        'grad': gradient,
        'conjugate_gradient': conjugate_gradient,
        'cg': conjugate_gradient,
        'gradiente': gradient,
        'gradiente_coniugato': conjugate_gradient
    }

    if method not in method_map:
        raise ValueError(
            f"Metodo {method} non valido. Scegli tra: \n"
            "- 'jacobi'\n"
            "- 'gauss_seidel'/'gs'\n"
            "- 'gradient'/'grad'\n"
            "- 'conjugate_gradient'/'cg'"
        )

    return method_map[method](A, b, x_exact, tol, max_iter)


def jacobi(A, b, x_exact, tol=1e-6, max_iter=20000):
    """
    Metodo di Jacobi per sistemi lineari
    """
    x = np.zeros_like(b)
    D = sp.diags(A.diagonal())
    D_inv = sp.diags(1 / A.diagonal())
    R = A - D
    start_time = process_time()

    for k in range(max_iter):
        x_new = D_inv.dot(b - R.dot(x))
        residual = A.dot(x_new) - b
        rel_res = np.linalg.norm(residual) / np.linalg.norm(b)

        if rel_res < tol:
            break
        x = x_new

    error = np.linalg.norm(x - x_exact) / np.linalg.norm(x_exact)
    return x, k + 1, process_time() - start_time, error


def gauss_seidel(A, b, x_exact, tol=1e-6, max_iter=20000):
    """
    Metodo di Gauss-Seidel per sistemi lineari
    """
    x = np.zeros_like(b)
    start_time = process_time()
    A = A.toarray()  # Per semplicità di implementazione

    for k in range(max_iter):
        x_old = x.copy()
        for i in range(A.shape[0]):
            x[i] = (b[i] - A[i, :i] @ x[:i] - A[i, i + 1:] @ x_old[i + 1:]) / A[i, i]

        residual = A.dot(x) - b
        rel_res = np.linalg.norm(residual) / np.linalg.norm(b)
        if rel_res < tol:
            break

    error = np.linalg.norm(x - x_exact) / np.linalg.norm(x_exact)
    return x, k + 1, process_time() - start_time, error


def gradient(A, b, x_exact, tol=1e-6, max_iter=20000):
    """
    Metodo del Gradiente per sistemi lineari
    """
    x = np.zeros_like(b)
    r = b - A.dot(x)
    p = r.copy()
    start_time = process_time()

    for k in range(max_iter):
        Ap = A.dot(p)
        alpha = r.dot(r) / (p.dot(Ap))
        x += alpha * p
        r_new = r - alpha * Ap

        rel_res = np.linalg.norm(r_new) / np.linalg.norm(b)
        if rel_res < tol:
            break

        beta = r_new.dot(r_new) / (r.dot(r))
        p = r_new + beta * p
        r = r_new

    error = np.linalg.norm(x - x_exact) / np.linalg.norm(x_exact)
    return x, k + 1, process_time() - start_time, error


def conjugate_gradient(A, b, x_exact, tol=1e-6, max_iter=20000):
    """
    Metodo del Gradiente Coniugato per sistemi lineari
    """
    x = np.zeros_like(b)
    r = b - A.dot(x)
    p = r.copy()
    rsold = r.dot(r)
    start_time = process_time()

    for k in range(max_iter):
        Ap = A.dot(p)
        alpha = rsold / (p.dot(Ap))
        x += alpha * p
        r -= alpha * Ap
        rsnew = r.dot(r)

        rel_res = np.sqrt(rsnew) / np.linalg.norm(b)
        if rel_res < tol:
            break

        p = r + (rsnew / rsold) * p
        rsold = rsnew

    error = np.linalg.norm(x - x_exact) / np.linalg.norm(x_exact)
    return x, k + 1, process_time() - start_time, error