"""
    FILE FUNCTIONS.PY 

    AUTORI: 
    - Biancini Mattia 865966
    - Gargiulo Elio 869184

    DESCRIZIONE:
    File principale del progetto.
    Contiene tutte le funzioni necessarie alla compressione dell'immagine,
    a partire dalla divisione in blocchi FxF, applicazione della dct2 ad ogni blocco,
    rimozione delle frequenze, idct2 e ricostruzione dell'immagine compressa.

"""

import numpy as np
from scipy.fft import dctn, idctn
from PIL import Image

# Valido la dimensione massima del blocco
def validate_dim(F, image_height=None, image_width=None):
    # Controllo se sostanzialmente ho un blocco maggiore della dim della immagine
    if image_height is not None and image_width is not None:
        if F > image_height or F > image_width:
            return False
    return True

# Validazione input
def validate_data(d, F):
    # Controllo se la d è corretta
    if 0 <= d <= 2 * F - 2:
        # Controllo se F è positivo
        if F > 0:
            return True
    else:
        return False

# Effettuo tutte le operazioni per comprimere l'img
def compress_image(image, F, d):
    # Ottengo l'immagine in formato matriciale
    image_matrix = np.array(image)
    # Salvo le dimensioni
    image_height, image_width = image_matrix.shape
    # Valido se ho blocchi corretti
    if not validate_dim(F, image_height, image_width):
        raise Exception("Errore: L'immagine sarà tutta nera e quindi vuota per la dimensione dei blocchi scelta. Scegliere un'altro F")
    # Lista per i blocchi finali
    ff_blocks = []
    # Ottengo i blocchi FxF
    blocks = get_blocks(image_matrix, image_height, image_width, F)
    # Itero sui blocchi
    for f in blocks:
        # Trasformo in array
        f = np.array(f)
        # Applico la dct2
        c = dct2(f)
        # Elimino le frequenze
        c_compressed = compress_block(c, d, F)
        # Applico la dct2 e casting al formato rgb
        ff = idct_block(c_compressed)
        # Aggiungo il blocco compresso alla lista
        ff_blocks.append(ff)
    # Ricostruisco l'immagine con i blocchi elaborati
    compressed_image = reconstruct_image(ff_blocks, image_width, image_height, F)
    # Ritorno l'immagine compressa
    return compressed_image

# Funzione per suddividere la matrice in blocchi F x F, scartando gli avanzi
def get_blocks(matrix, image_height, image_width, F):
    blocks = []
    # Calcola il numero di blocchi verticali e orizzontali
    num_blocks_v = image_height // F
    num_blocks_h = image_width // F
    # Intero sul numero di blocchi
    for i in range(num_blocks_v):
        for j in range(num_blocks_h):
            # Calcola gli indici di inizio e fine per le righe e le colonne del blocco
            start_row = i * F
            end_row = (i + 1) * F
            start_col = j * F
            end_col = (j + 1) * F
            # Estrae il blocco F x F
            block = matrix[start_row:end_row, start_col:end_col]
            blocks.append(block)
    
    return blocks

# Elimino le componenti di un blocco ckl con k+l ≥ d
def compress_block(block, d, F):
    # Inizializzo il blocco "tagliato"
    compressed_block = np.zeros_like(block)
    # Itero su k e l a partire da 0 fino a lungh del blocco F
    for k in range(F):
        for l in range(F):
            # Salvo il risultato di k + l
            keep = k + l
            # Se keep e' minore di d, allora la quella frequenza
            if keep < d:
                compressed_block[k][l] = block[k][l]
    # Ritorno i blocchi "tagliati"
    return compressed_block

# Applico idct2 per tornare all'img, arrotondando le componenti
# con valori corretti per rgb a 1 byte
def idct_block(compressed_block):
    # Applico la IDCT2 della libreria
    ff = idct2(compressed_block)
    # Arrotondo il blocco ad intero
    ff_rounded = np.round(ff)
    # Mi assicuro che il blocco sia nel range (0,255)
    ff_valid = np.clip(ff_rounded, 0, 255)
    # Converto ad un byte
    ff_byte = ff_valid.astype(np.uint8)
    # Ritorno il blocco
    return ff_byte

# Funzione per ricostruire l'immagine dai blocchi F x F
def reconstruct_image(blocks, image_width, image_height, F):

     # Calcola il numero di blocchi verticali e orizzontali
    num_blocks_v = image_height // F
    num_blocks_h = image_width // F
    # Crea una nuova matrice per l'immagine
    new_image_height = F * num_blocks_v
    new_image_width = F * num_blocks_h
    # Nuova immagine
    new_image_matrix = np.zeros((new_image_height, new_image_width), dtype=np.uint8)
    # Itero sui blocchi
    for i in range(num_blocks_v):
        for j in range(num_blocks_h):
            # Calcola gli indici di inizio e fine per le righe e le colonne del blocco
            start_row = i * F       # Inizio riga blocco i
            end_row = (i + 1) * F   # Fine riga blocco i
            start_col = j * F       # Inizio colonna blocco j
            end_col = (j + 1) * F   # Fine colonna blocco j
            # Assegna il blocco F x F alla nuova matrice dell'immagine
            # Abbiamo i blocchi salvati in ordine una lista quindi per averli possiamo fare:
            # blocco 1 = riga 0 * blocchi in orizzontale + colonna 1 = 1 etc
            new_image_matrix[start_row:end_row, start_col:end_col] = blocks[i * num_blocks_h + j]
    return Image.fromarray(new_image_matrix)

# DCT2 della libreria
def dct2(matrix):
    return dctn(matrix, norm='ortho')

# IDCT2 della libreria
def idct2(matrix):
    return idctn(matrix, norm='ortho')
