import numpy as np
import scipy.io as spio
from datetime import datetime
from methods import solve_linear_system
import os


def load_matrices(matrix_dir='matrix'):
    """Carica tutte le matrici .mtx dalla cartella specificata"""
    matrices = {}
    for file in os.listdir(matrix_dir):
        if file.endswith('.mtx'):
            name = file.split('.')[0]
            path = os.path.join(matrix_dir, file)
            matrices[name] = spio.mmread(path).tocsr()  # Converti in formato sparso CSR
    return matrices


def create_problem(A):
    """Crea il problema Ax=b con soluzione esatta x=[1,1,...,1]"""
    n = A.shape[0]
    x_exact = np.ones(n)
    b = A.dot(x_exact)
    return b, x_exact


def run_experiments(matrices, tolerances):
    """Esegue tutti gli esperimenti e raccoglie i risultati"""
    results = {}

    for mat_name, A in matrices.items():
        print(f"\n{'=' * 40}\nProcessing matrix: {mat_name}\n{'=' * 40}")
        b, x_exact = create_problem(A)
        results[mat_name] = {}

        for tol in tolerances:
            print(f"\n▶ Tolleranza: {tol:.0e}")
            current_results = {}

            # Lista dei metodi da testare
            methods = [
                'jacobi',
                'gauss_seidel',
                'gradient',
                'conjugate_gradient'
            ]

            for method in methods:
                start_time = datetime.now()
                x, iterations, time, error = solve_linear_system(
                    method=method,
                    A=A,
                    b=b,
                    x_exact=x_exact,
                    tol=tol
                )
                exec_time = (datetime.now() - start_time).total_seconds()

                current_results[method] = {
                    'iterations': iterations,
                    'time': exec_time,
                    'error': error,
                    'residual': np.linalg.norm(A.dot(x) - b) / np.linalg.norm(b)
                }

            results[mat_name][tol] = current_results

    return results


def print_results(results):
    """Stampa i risultati in formato tabellare"""
    for mat_name, mat_results in results.items():
        print(f"\n{'#' * 60}")
        print(f"RISULTATI PER LA MATRICE: {mat_name.upper()}")
        print(f"{'#' * 60}")

        for tol, tol_results in mat_results.items():
            print(f"\n▪ Tolleranza: {tol:.0e}")
            print("-" * 55)
            print(f"{'Metodo':<20} | {'Iterazioni':<10} | {'Tempo (s)':<10} | {'Errore Relativo':<15}")
            print("-" * 55)

            for method, data in tol_results.items():
                print(f"{method:<20} | {data['iterations']:<10} | {data['time']:<10.4f} | {data['error']:<15.2e}")
            print("-" * 55)


if __name__ == "__main__":
    # Configurazione
    matrix_dir = 'matrix'  # Cartella contenente le matrici
    tolerances = [1e-4, 1e-6, 1e-8,1e-10]  # Lista di tolleranze da testare

    # Caricamento matrici
    print("Caricamento matrici...")
    matrices = load_matrices(matrix_dir)

    # Esecuzione esperimenti
    print("\nAvvio esperimenti...")
    start_time = datetime.now()
    results = run_experiments(matrices, tolerances)
    total_time = (datetime.now() - start_time).total_seconds()

    # Stampa risultati
    print("\n\n" + "=" * 60)
    print(f"RIEPILOGO FINALE - Tempo totale esecuzione: {total_time:.2f} secondi")
    print("=" * 60)
    print_results(results)