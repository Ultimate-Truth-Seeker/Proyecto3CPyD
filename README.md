# Proyecto 3 — Mejorando algoritmos con CUDA (CC3069, UVG)

Aceleración del **Laboratorio 3 de Data Science (reconocimiento de lenguaje de señas ASL)** separando el trabajo:

- **CPU (multi-proceso):** lectura de JPG, resize y normalización de las imágenes.
- **GPU (CUDA/cuDNN/cuBLAS):** entrenamiento e inferencia de la CNN 2 del Lab 3.

Se mide el tiempo secuencial (1 proceso + entrenamiento en CPU) contra el paralelo y se calcula el speedup por fase.

## Contenido
| Ruta | Qué es |
|---|---|
| `notebooks/proyecto3_cuda.ipynb` | Experimento principal (ya ejecutado en Colab, con salidas): código original secuencial + versión paralela, mediciones, monitor de recursos, tablas y gráficas. |
| `notebooks/proyecto3_complemento.ipynb` | Mediciones complementarias en Colab: kernels CUDA propios (Numba y CUDA C), costo de transferencia CPU↔GPU y entrenamiento hasta convergencia. |
| `src/cpu_prep.py` | Worker de la preparación paralela (el notebook lo regenera; se versiona por claridad). |
| `docs/informe_rendimiento.pdf` / `.md` | Informe final. Se regenera con `python docs/build_informe.py .` a partir de los CSV de `results/` (las secciones del complemento se llenan solas al agregar sus CSV). |
| `results/` | CSV, gráficas, logs de `nvidia-smi` y capturas generados al ejecutar el notebook. |

## Cómo reproducir
1. Abrir `notebooks/proyecto3_cuda.ipynb` en Colab (`Archivo → Subir cuaderno` o desde GitHub).
2. `Entorno de ejecución → Cambiar tipo de entorno → GPU (T4)`.
3. `Ejecutar todo`. La primera celda de datos descarga el dataset de Kaggle con `kagglehub` (si pide credenciales, usar `kagglehub.login()`).
4. Mientras corre cada fase, tomar capturas de `nvidia-smi` / panel *Recursos* de Colab.
5. El notebook descarga `resultados_proyecto3.zip`; descomprimirlo dentro de `results/`. Repetir con `notebooks/proyecto3_complemento.ipynb` (`resultados_complemento.zip`) y regenerar el informe.

Parámetros en la segunda celda de código: `EPOCHS_BENCH` (épocas fijas), `REPEATS` (repeticiones), `N_PER_CLASS`.

## Decisiones de diseño
- Mismas rutas de imagen, mismo split y mismas semillas en secuencial y paralelo; la versión paralela se verifica **idéntica bit a bit** (`np.array_equal`).
- Épocas fijas, sin `EarlyStopping`, para que CPU y GPU hagan exactamente el mismo trabajo.
- Calentamiento (warm-up) sin medir y reporte aparte de la primera época de GPU.
- Las transferencias CPU↔GPU están incluidas en el tiempo de `fit()` y además se miden aparte.
