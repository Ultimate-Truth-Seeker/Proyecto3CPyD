"""
src/cpu_prep.py
Proyecto 3 - CC3069 Computación Paralela y Distribuida (UVG)

Preparación de imágenes en CPU con múltiples procesos.

La decodificación JPEG + resize + conversión a RGB es trabajo con ramas y acceso a
disco (no es una operación regular tipo matriz), por eso se reparte entre los núcleos
del CPU con un pool de procesos y NO se manda al GPU.

Este módulo se mantiene aparte y liviano (solo PIL + numpy) porque los procesos se
crean con el método "spawn": cada worker importa este archivo sin arrastrar TensorFlow
ni el contexto CUDA del proceso principal (hacer fork con CUDA/TF inicializados puede
colgar el proceso).
"""

import numpy as np
from PIL import Image


def load_one(args):
    """
    Carga una imagen, la pasa a RGB y la redimensiona. Devuelve uint8 (H, W, 3).
    Es EXACTAMENTE la misma operación de PIL que usa la versión secuencial
    (convert("RGB").resize((s, s))); la normalización a [0, 1] se hace afuera,
    vectorizada, para que el resultado sea idéntico bit a bit y para enviar 4 veces
    menos bytes entre procesos (uint8 en vez de float32).
    """
    path, img_size = args
    with Image.open(path) as im:
        return np.asarray(im.convert("RGB").resize((img_size, img_size)), dtype=np.uint8)
