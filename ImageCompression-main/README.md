# Image Compression


> Methods of Scientific Computing Project for University Milano Bicocca. 2023-2024. Grade: 30

[![Download Relazione PDF](https://img.shields.io/badge/Download%20Relazione-PDF-lime.svg?style=for-the-badge)](https://github.com/Zeptogram/ImageCompression/releases/download/final/MCS_Relazione_Progetto_2_Biancini_Mattia_865966_Gargiulo_Elio_869184.Final.pdf)
[![Download Presentazione PDF](https://img.shields.io/badge/Download%20Presentazione-PDF-orange.svg?style=for-the-badge)](https://github.com/Zeptogram/ImageCompression/releases/download/final/Progetto.2.MCS.-.Presentazione.pdf)
[![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)](https://www.python.org/)

DTC2 Implementation and JPEG-like Compression of grayscale images.

## Project Overview

The project is divided into two sub-projects corresponding to the two parts of the assignment and a directory for the images:

- **imgs** » Directory containing the images provided with the assignment.
- **part1** » Sub-project related to the first part of the assignment:
  - Implementation of DCT2
  - Analysis of the DCT2 provided by SciPy and comparison with our version
- **part2** » Sub-project related to the second part of the assignment:
  - Import of a grayscale image
  - Reconstruction of the compressed image after processing

Each project consists of two files:

- `functions.py` which collects all the functions we used as if it were a library for our project
- `main.py` executable of the related sub-project

Download the documentation for detailed info about the whole project (only in Italian). 

## Authors

- Mattia Biancini
- Elio Gargiulo

## Useful Resources
- SciPy DCT: [docs](https://docs.scipy.org/doc/scipy/reference/generated/scipy.fftpack.dct.html)
- DCT: [wiki](https://en.wikipedia.org/wiki/Discrete_cosine_transform)
- JPEG Compression: [wiki](https://en.wikipedia.org/wiki/JPEG)
