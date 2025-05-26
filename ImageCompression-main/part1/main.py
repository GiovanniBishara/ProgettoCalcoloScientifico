import numpy as np
import functions as dct

# Per testing
testing = False

if testing:
    test_vector = np.array([231, 32, 233, 161, 24, 71, 140, 245])
    test_matrix = np.array([[231, 32, 233, 161, 24, 71, 140, 245],
                            [247, 40, 248, 245, 124, 204, 36, 107],
                            [234, 202, 245, 167, 9, 217, 239, 173],
                            [193, 190, 100, 167, 43, 180, 8, 70],
                            [11, 24, 210, 177, 81, 243, 8, 112],
                            [97, 195, 203, 47, 125, 114, 165, 181],
                            [193, 70, 174, 167, 41, 30, 127, 245],
                            [87, 149, 57, 192, 65, 129, 178, 228]])
    # Salvo i risultati delle dct
    custom_dct_res =  dct.dct_custom(test_vector)
    custom_dct2_res = dct.dct2_custom(test_matrix)
    library_dct_res = dct.dct_library(test_vector)
    library_dct2_res = dct.dct2_library(test_matrix)
    # Scrittura su file delle dct
    dct.write_results_to_file("DCT CUSTOM", custom_dct_res)
    dct.write_results_to_file("DCT LIBRARY", library_dct_res)
    dct.write_results_to_file("DCT2 CUSTOM", custom_dct2_res)
    dct.write_results_to_file("DCT2 LIBRARY", library_dct2_res)
    # Test di equality
    dct_eq = np.allclose(custom_dct_res, library_dct_res)
    print("Le DCT sono uguali: ", dct_eq)
    dct2_eq = np.allclose(custom_dct2_res, library_dct2_res)
    print("Le DCT2 sono uguali: ", dct2_eq)
else:
    # Analisi sulle DCT2
    dct.dct2_analyze_graph()