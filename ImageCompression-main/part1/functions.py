import numpy as np
from scipy.fft import dct, dctn
import os
import matplotlib.pyplot as plt
import timeit

# Per debugging
debug = True
show_basis = False

def dct_custom(vector):
    # Inizializzo le variabili
    n = len(vector)
    pi = np.pi
    dct = np.zeros(n)
    wk = np.zeros(n)
    cos_base = []  
    # K = 0 a n-1
    for k in range(n):
        # Vettori della base dei coseni
        for i in range(n):
            # Calcolo le componenti wki come cos(pi * k * (2 * i + 1) / (2 * n))
            wk[i] = np.cos(pi * k * (2 * i + 1) / (2 * n))
        # Normalizzazione ortho per confronto con la dct della libreria
        if k == 0:
            # Calcolo i coefficenti ak = (v * wk) / (wk * wk)
            ak = np.dot(vector, wk) / np.sqrt(n)   # wk * wk
        else:
            ak = np.dot(vector, wk) / np.sqrt(n/2) # wk * wk
        # Salvo la base se debug
        if show_basis:
            cos_base.append(wk.copy())
        # Aggiungo alla lista gli ak
        dct[k] = ak
    # Plotto i coseni per freq k
    if show_basis:
        plot_cosine_base(cos_base)
    return dct

def dct2_custom(matrix):
    # Creo una matrice di appoggio
    dct_matrix_row = np.zeros_like(matrix, dtype=float)
    # Applico la DCT alle righe della matrice
    for i in range(len(matrix)):
        dct_matrix_row[i, :] = dct_custom(matrix[i, :])
    # Applico la DCT alle colonne della matrice dct righe
    dct_matrix_col = dct_matrix_row.T
    for i in range(len(dct_matrix_col)):
        dct_matrix_col[i, :] = dct_custom(dct_matrix_col[i, :])
    # Rifaccio la trasposta e ritorno la matrice
    return dct_matrix_col.T


# DCT di SciPy
def dct_library(vector):
    # Equivalente a dct di matlab
    return dct(vector, 2, norm='ortho')


# DCT2 di SciPy
def dct2_library(matrix):
    return dctn(matrix, 2, norm='ortho')


# Funzione per analizzare i tempi di esecuzione e generare il grafico
def dct2_analyze_graph():
    matrices, N = generate_square_matrices(40, 5)
    # Calcolo i tempi di esecuzione per le due dct2
    dct2_custom_times = measure_dct2_times(matrices, dct2_custom, True)
    dct2_library_times = measure_dct2_times(matrices, dct2_library, False)
    # Scrivo su file i tempi
    if debug:
        write_times_to_file(dct2_custom_times, dct2_library_times)
    # Curve teoriche per il confronto
    N_cubed = N ** 3
    N_squared_logN = N ** 2 * np.log(N)
    # Normalizzare le curve teoriche per confrontarle con i tempi reali
    N_cubed_normalized = N_cubed / N_cubed[-1] * dct2_custom_times[-1]
    N_squared_logN_normalized = N_squared_logN / N_squared_logN[-1] * dct2_library_times[-1]
    # Creazione del grafico
    plt.figure(figsize=(10, 6))
    plt.semilogy(N, dct2_custom_times, 'o-', label='DCT2 Custom')
    plt.semilogy(N, dct2_library_times, 's-', label='DCT2 Libreria')
    # Curve teoriche (tratteggiate)
    plt.semilogy(N, N_cubed_normalized, 'k--', label='$N^3$')
    plt.semilogy(N, N_squared_logN_normalized, 'r--', label='$N^2 \log N$')
    # Labels
    plt.xlabel('Dimensione N')
    plt.ylabel('Tempo di esecuzione (s)')
    plt.title('Confronto Tempi di Esecuzione DCT2')
    plt.legend()
    plt.grid(True)
    plt.xticks(N)
    # Mostra il grafico
    plt.show()


# UTILITA'

# Genera matrici NxN casuali di dimensione start_size fino a num_doublings raddoppiando in volta
def generate_square_matrices(start_size, num_doublings):
    sizes = [start_size * 2 ** i for i in range(num_doublings)]
    matrices = [np.random.rand(size, size) for size in sizes]
    return matrices, np.array(sizes)



# Misura i tempi di esecuzione della DCT2 custom per ogni matrice NxN
def measure_dct2_times(matrices, function, custom=True):
    times = []
    for matrix in matrices:
        # Se debug
        if debug:
            if custom:
                print("[LOG DCT2 CUSTOM]: Eseguendo la DCT2 sulla Matrice {} x {}".format(matrix.shape[0], matrix.shape[1]))
            else:
                print("[LOG DCT2 LIBRARY]: Eseguendo la DCT2 sulla Matrice {} x {}".format(matrix.shape[0], matrix.shape[1]))
        # Timeit per un'ottima precisione
        time_taken = timeit.timeit(lambda: function(matrix), number=1)
        times.append(time_taken)
    return times

# DEBUG

# Per debugging stampa delle basi dei coseni per vedere se sono corrette
def plot_cosine_base(base):
    n = len(base[0])  # Lunghezza della base
    num_base = len(base)
    ncols = 3  # Numero di colonne nella griglia
    nrows = -(-num_base // ncols)  # Calcolo del numero di righe
    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 8))
    for i, wk in enumerate(base):
        row = i // ncols
        col = i % ncols
        axes[row, col].bar(range(n), wk, label=f'k={i}')
        axes[row, col].set_xlabel('Indice i fino a n')
        axes[row, col].set_ylabel('Valore')
        axes[row, col].set_title(f'Basi dei Coseni, k={i} oscillazioni')
        axes[row, col].legend()
    plt.tight_layout()
    plt.show()


def write_results_to_file(dtype, result):
    # Verifica se il file "dct_results" esiste già
    file_name = "dct_results.txt"
    file_exists = os.path.exists(file_name)
    # Se il file non esiste, crea un nuovo file "dct_results.txt"
    if not file_exists:
        with open(file_name, "w") as file:
            file.write("====[ RISULTATI DCT E DCT2 ]====\n\n")
    # Scrivi i dati nella tupla nel file
    with open(file_name, "a") as file:
        file.write('=== {} ===\n'.format(dtype))
        file.write('Result: {}\n'.format(result))
        file.write("")


def write_times_to_file(dct2_custom_times, dct2_library_times):
    # Verifica se il file "dct_times" esiste già
    file_name = "dct_times.txt"
    file_exists = os.path.exists(file_name)
    # Se il file non esiste, crea un nuovo file "dct_times.txt"
    if not file_exists:
        with open(file_name, "w") as file:
            file.write("====[ TEMPI DCT2 CUSTOM E LIBRARY ]====\n\n")
    # Scrivi i dati nella tupla nel file
    with open(file_name, "a") as file:
        file.write('=== { DCT2 CUSTOM } ===\n')
        for i in range(len(dct2_custom_times)):
            file.write('{}\n'.format(dct2_custom_times[i]))
        file.write('=== { DCT2 LIBRARY } ===\n')
        for i in range(len(dct2_library_times)):
            file.write('{}\n'.format(dct2_library_times[i]))