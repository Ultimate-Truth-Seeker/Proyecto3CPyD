## Entorno

```
Sat Nov 14 04:53:30 2026       
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 580.82.07              Driver Version: 580.82.07      CUDA Version: 13.0     |
+-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  Tesla T4                       Off |   00000000:00:04.0 Off |                    0 |
| N/A   42C    P8              9W /   70W |       3MiB /  15360MiB |      0%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+

+-----------------------------------------------------------------------------------------+
| Processes:                                                                              |
|  GPU   GI   CI              PID   Type   Process name                        GPU Memory |
|        ID   ID                                                               Usage      |
|======================================================================================
```

- Núcleos CPU: 2  |  TensorFlow 2.21.0  |  Épocas de benchmark: 4  |  Repeticiones: 3


## Tiempos de preparación (s)

| Ejecución | Secuencial | Paralelo |
|---|---|---|
| 1 | 23.14 | 20.40 |
| 2 | 23.46 | 20.29 |
| 3 | 23.96 | 20.14 |
| **Promedio ± σ** | 23.52 ± 0.34 | 20.28 ± 0.11 |


## Entrenamiento / inferencia (promedio de las ejecuciones)

| config                              |   fit_s |   fit_std |   epoca_estable_s |   primera_epoca_s |   inferencia_s |   inferencia_std |   acc_test |
|:------------------------------------|--------:|----------:|------------------:|------------------:|---------------:|-----------------:|-----------:|
| CPU (secuencial, bs=32)             | 692.195 |    38.243 |           175.849 |           163.739 |          8.229 |            1.948 |      0.718 |
| GPU (CUDA, bs=32)                   |  31.504 |     5.38  |             3.695 |            18.87  |          3.189 |            1.655 |      0.686 |
| GPU optimizado (CUDA, bs=128, fp16) |  26.787 |     4.212 |             1.746 |            19.97  |          2.486 |            0.694 |      0.206 |


## Speedup por fase

| Fase                                                               |   Secuencial (s) |   Paralelo (s) |   Speedup |   Eficiencia |
|:-------------------------------------------------------------------|-----------------:|---------------:|----------:|-------------:|
| A. Preparación de imágenes (CPU 1 proceso → CPU 2 procesos)        |           23.52  |         20.277 |     1.16  |         0.58 |
| B. Entrenamiento (4 épocas) — GPU (CUDA, bs=32)                    |          692.195 |         31.504 |    21.972 |       nan    |
| B'. Época en régimen estable — GPU (CUDA, bs=32)                   |          175.849 |          3.695 |    47.591 |       nan    |
| C. Inferencia sobre test — GPU (CUDA, bs=32)                       |            8.229 |          3.189 |     2.58  |       nan    |
| Extremo a extremo (A+B+C) — GPU (CUDA, bs=32)                      |          723.945 |         54.97  |    13.17  |       nan    |
| B. Entrenamiento (4 épocas) — GPU optimizado (CUDA, bs=128, fp16)  |          692.195 |         26.787 |    25.841 |       nan    |
| B'. Época en régimen estable — GPU optimizado (CUDA, bs=128, fp16) |          175.849 |          1.746 |   100.724 |       nan    |
| C. Inferencia sobre test — GPU optimizado (CUDA, bs=128, fp16)     |            8.229 |          2.486 |     3.311 |       nan    |
| Extremo a extremo (A+B+C) — GPU optimizado (CUDA, bs=128, fp16)    |          723.945 |         49.55  |    14.61  |       nan    |


## Escalamiento de la fase A

|   procesos |   tiempo_s |   speedup_vs_secuencial |
|-----------:|-----------:|------------------------:|
|          1 |     23.925 |                   0.983 |
|          2 |     20.219 |                   1.163 |
|          4 |     19.281 |                   1.22  |
|          8 |     21.552 |                   1.091 |
