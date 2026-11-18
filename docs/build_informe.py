"""Genera docs/informe_rendimiento.md (+ PDF) a partir de results/*.csv.
Uso: python build_report.py <repo_dir> [--pdf]
Las secciones que dependen del notebook complementario se llenan solas si existen sus CSV."""
import sys, subprocess
from pathlib import Path
import numpy as np, pandas as pd

repo = Path(sys.argv[1]); R = repo / "results"
runs = pd.read_csv(R / "benchmark_ejecuciones.csv")
summ = pd.read_csv(R / "benchmark_resumen.csv")
swp = pd.read_csv(R / "prep_escalamiento.csv")
seq_t = [23.14, 23.46, 23.96]; par_t = [20.40, 20.29, 20.14]     # de results/informe_tablas.md (salida del notebook)
N_CPU = 2

def g(cfg_prefix, col): return summ[summ.config.str.startswith(cfg_prefix)][col].iloc[0]
cpu = {c: g("CPU", c) for c in ["fit_s", "fit_std", "epoca_estable_s", "primera_epoca_s", "inferencia_s", "inferencia_std", "acc_test"]}
g32 = {c: g("GPU (CUDA, bs=32", c) for c in ["fit_s", "fit_std", "epoca_estable_s", "primera_epoca_s", "inferencia_s", "inferencia_std", "acc_test"]}
g128 = {c: g("GPU optimizado", c) for c in ["fit_s", "fit_std", "epoca_estable_s", "primera_epoca_s", "inferencia_s", "inferencia_std", "acc_test"]}
ps, pp = np.mean(seq_t), np.mean(par_t)
e2e_seq = ps + cpu["fit_s"] + cpu["inferencia_s"]
e2e32 = pp + g32["fit_s"] + g32["inferencia_s"]; e2e128 = pp + g128["fit_s"] + g128["inferencia_s"]

def usage(name):
    d = pd.read_csv(R / f"uso_{name}.csv"); cores = [c for c in d.columns if c.startswith("cpu") and c != "cpu_mean"]
    return dict(dur=d.t.max(), cpu=d.cpu_mean.mean(), cpu_max=d.cpu_mean.max(), cores=[d[c].mean() for c in cores],
                gpu=d.gpu_util.mean(), gpu_max=d.gpu_util.max(), gpu_p90=d.gpu_util.quantile(.9), mem=d.gpu_mem_used_mb.max(), ram=d.ram_used_gb.max())
U = {n: usage(n) for n in ["A_prep_secuencial", "A_prep_paralelo_CPU", "B_entrenamiento_CPU", "B_entrenamiento_GPU", "B_entrenamiento_GPU_optimizado"]}

def f(x, d=2): return f"{x:.{d}f}"
def tbl(df, fmt=None):
    cols = list(df.columns); out = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for _, r in df.iterrows(): out.append("| " + " | ".join(str(r[c]) for c in cols) + " |")
    return "\n".join(out)

# ---------------- tablas ----------------
t_prep = pd.DataFrame({"Ejecución": [1, 2, 3, "**Promedio ± σ**"],
                       "Secuencial, 1 proceso (s)": [f(x) for x in seq_t] + [f"**{f(ps)} ± {f(np.std(seq_t))}**"],
                       f"Paralelo, {N_CPU} procesos (s)": [f(x) for x in par_t] + [f"**{f(pp)} ± {f(np.std(par_t))}**"]})
rr = runs.copy()
rr["Configuración"] = rr["config"]; rr["Ejec."] = rr["run"]
t_runs = rr[["Configuración", "Ejec.", "train_s", "epoch_first_s", "epoch_steady_s", "infer_s", "test_acc"]].rename(columns={
    "train_s": "fit 4 épocas (s)", "epoch_first_s": "1.ª época (s)", "epoch_steady_s": "época estable (s)", "infer_s": "inferencia (s)", "test_acc": "acc. test"})
for c in ["fit 4 épocas (s)", "1.ª época (s)", "época estable (s)", "inferencia (s)"]: t_runs[c] = t_runs[c].map(lambda x: f(x, 1) if x > 5 else f(x, 2))
t_runs["acc. test"] = t_runs["acc. test"].map(lambda x: f(100 * x, 1) + " %")
t_sum = pd.DataFrame({
    "Configuración": ["CPU (secuencial, bs=32)", "GPU CUDA (bs=32)", "GPU CUDA optimizado (bs=128, fp16)"],
    "fit 4 épocas (s)": [f"{f(d['fit_s'],1)} ± {f(d['fit_std'],1)}" for d in (cpu, g32, g128)],
    "época estable (s)": [f(d["epoca_estable_s"]) for d in (cpu, g32, g128)],
    "1.ª época (s)": [f(d["primera_epoca_s"], 1) for d in (cpu, g32, g128)],
    "inferencia test (s)": [f"{f(d['inferencia_s'])} ± {f(d['inferencia_std'])}" for d in (cpu, g32, g128)],
    "acc. test (media)": [f(100 * d["acc_test"], 1) + " %" for d in (cpu, g32, g128)]})
sp = lambda a, b: f(a / b, 1) + "×"
t_speed = pd.DataFrame({
    "Fase": ["A. Preparación de imágenes (1 → 2 procesos en CPU)", "B. Entrenamiento, 4 épocas (CPU → GPU)", "B′. Época en régimen estable (CPU → GPU)",
             "C. Inferencia sobre test (CPU → GPU)", "**Extremo a extremo A+B+C**"],
    "Secuencial (s)": [f(ps), f(cpu["fit_s"], 1), f(cpu["epoca_estable_s"], 1), f(cpu["inferencia_s"]), f"**{f(e2e_seq, 1)}**"],
    "GPU bs=32 (s)": [f(pp), f(g32["fit_s"], 1), f(g32["epoca_estable_s"], 2), f(g32["inferencia_s"]), f"**{f(e2e32, 1)}**"],
    "Speedup bs=32": [sp(ps, pp), sp(cpu["fit_s"], g32["fit_s"]), sp(cpu["epoca_estable_s"], g32["epoca_estable_s"]), sp(cpu["inferencia_s"], g32["inferencia_s"]), f"**{sp(e2e_seq, e2e32)}**"],
    "GPU bs=128+fp16 (s)": [f(pp), f(g128["fit_s"], 1), f(g128["epoca_estable_s"], 2), f(g128["inferencia_s"]), f"**{f(e2e128, 1)}**"],
    "Speedup bs=128+fp16": [sp(ps, pp), sp(cpu["fit_s"], g128["fit_s"]), sp(cpu["epoca_estable_s"], g128["epoca_estable_s"]), sp(cpu["inferencia_s"], g128["inferencia_s"]), f"**{sp(e2e_seq, e2e128)}**"]})
t_sw = swp.assign(**{"Procesos": swp.procesos, "Tiempo (s)": swp.tiempo_s.map(lambda x: f(x)), "Speedup vs. secuencial": swp.speedup_vs_secuencial.map(lambda x: f(x))})[["Procesos", "Tiempo (s)", "Speedup vs. secuencial"]]
u = U
t_use = pd.DataFrame({
    "Fase": ["A secuencial (1 proceso)", f"A paralela ({N_CPU} procesos)", "B entrenamiento CPU", "B entrenamiento GPU (bs=32)", "B entrenamiento GPU (bs=128, fp16)"],
    "CPU medio (%)": [f(u[k]["cpu"], 0) for k in U],
    "CPU por vCPU (%)": [" / ".join(f(c, 0) for c in u[k]["cores"]) for k in U],
    "GPU util. media / p90 / máx (%)": [f"{f(u[k]['gpu'],0)} / {f(u[k]['gpu_p90'],0)} / {f(u[k]['gpu_max'],0)}" for k in U],
    "Memoria GPU máx. (MB)": [f(u[k]["mem"], 0) for k in U]})

# ---------------- secciones opcionales (complemento) ----------------
PEND = "> ⏳ **Pendiente:** ejecutar `notebooks/proyecto3_complemento.ipynb` en Colab, descomprimir `resultados_complemento.zip` en `results/` y regenerar este informe (`python docs/build_informe.py .`)."
def opt(name):
    p = R / name; return pd.read_csv(p) if p.exists() else None
kdf, trdf, cvdf = opt("complemento_kernels.csv"), opt("complemento_transferencia.csv"), opt("complemento_convergencia.csv")

if trdf is not None:
    t = trdf.assign(**{"Medición": trdf.metodo, "Tiempo (ms)": (trdf.s_mean * 1e3).map(lambda x: f(x, 1)) + " ± " + (trdf.s_std * 1e3).map(lambda x: f(x, 1)), "Ancho de banda (GB/s)": trdf.GBps.map(lambda x: f(x, 1))})[["Medición", "Tiempo (ms)", "Ancho de banda (GB/s)"]]
    h2d = trdf[trdf.metodo.str.startswith("Host→GPU (memoria paginada)")].iloc[0]
    sec_tr = f"Copia de {f(trdf.GB.iloc[0], 2)} GB (tamaño del conjunto de entrenamiento en `float32`), con sincronización explícita:\n\n{tbl(t)}\n\nLa copia completa del conjunto de entrenamiento cuesta ≈ {f(h2d.s_mean, 2)} s, es decir ≈ {f(100*h2d.s_mean/g32['fit_s'],1)} % del `fit` de 4 épocas en GPU (bs=32), y se paga una vez por época si los datos se envían por batch. Es chica frente al cómputo, pero no nula. La memoria *pinned* es {f(h2d.s_mean/trdf[trdf.metodo.str.contains('pinned')].s_mean.iloc[0],1)}× más rápida que la paginada y alcanza ≈ {f(trdf[trdf.metodo.str.contains('pinned')].GBps.iloc[0],1)} GB/s, cerca del máximo teórico del PCIe 3.0 ×16; traer datos de vuelta al *host* es más lento ({f(trdf[trdf.metodo.str.startswith('GPU→Host')].GBps.iloc[0],1)} GB/s), motivo para mantener los resultados en el GPU hasta el final y mover solo lo necesario. Como `fit()` envía los datos desde NumPy (memoria paginada), usar `tf.data` con `prefetch` o datos residentes en GPU sería una mejora posible."
else:
    sec_tr = ("La medición de la corrida principal (celda 6.3) **no es válida**: reportó 0.001 s (605 GB/s), una cifra imposible para el PCIe de una T4 (≈ 12 GB/s); `tf.identity` sobre un tensor ya residente no copió datos. "
              "En esa corrida el costo de transferencia está incluido dentro del tiempo de `fit()`, que alimenta los batches desde NumPy.\n\n" + PEND)

if kdf is not None:
    k = kdf.copy(); k["ms"] = k.ms.astype(float)
    k["Speedup vs. NumPy"] = (k.loc[0, "ms"] / k.ms).map(lambda x: f(x, 1) + "×")
    k["Tiempo (ms)"] = k.ms.map(lambda x: f(x, 3)); k["Correcto"] = k.correcto.map(lambda x: "✓" if x else "✗")
    k["Implementación"] = k.implementacion
    sec_k = tbl(k[["Implementación", "Tiempo (ms)", "Speedup vs. NumPy", "Correcto"]])
    nb = "Numba" in set(k.backend)
    sec_k += ("\n\n" + ("Los kernels de Numba y los de CUDA C producen el mismo resultado que NumPy." if nb else
              "Numba no pudo compilar los kernels en Colab (error de `nvJitLink`: el nvJitLink de CUDA 12.5 que trae TensorFlow es más viejo que el NVVM de Numba), por lo que se validó el mismo diseño de hilos/bloques/grid en CUDA C compilado con NVRTC desde CuPy.") +
              " En CUDA C el kernel con memoria compartida es " + sp(float(k.loc[3,'ms']), float(k.loc[4,'ms'])) + " más rápido que el ingenuo (cada dato cargado se reutiliza 16 veces dentro del bloque), y cuBLAS, afinado por NVIDIA, sigue siendo la referencia (" + sp(float(k.loc[4,'ms']), float(k.loc[5,'ms'])) + " más rápido que el mejor kernel propio)." +
              ("\n\n**Los kernels de Numba resultaron más lentos que los de CUDA C con el mismo diseño (≈ 3.5× el ingenuo y ≈ 7.6× el *tiled*) y el *tiling* no aportó nada** (" + f(float(k.loc[1,'ms']),1) + " ms vs. " + f(float(k.loc[2,'ms']),1) + " ms). Causa probable, no verificada con un perfilador: en Numba el acumulador se inicializó con `acc = 0.0`, que es `float64`, por lo que el producto-acumulación se hace en doble precisión, y la T4 tiene un rendimiento de `float64` ≈ 1/32 del de `float32`; ese costo aritmético domina y oculta la ventaja de la memoria compartida. La corrección esperada es inicializar con `float32(0.0)`. El NumPy de esta tabla (≈ 182 ms) es más lento que el de la corrida principal (70.7 ms) probablemente porque Colab asignó otra máquina o distinto número de hilos BLAS (no se verificó); por eso los speedups de la tabla son relativos a esta misma corrida." if nb else ""))
else:
    sec_k = ("En la corrida principal los kernels de Numba **no se pudieron ejecutar**: la compilación falló con `nvJitLink: ERROR_INTERNAL … may need newer version of nvJitLink library` "
             "(incompatibilidad entre el nvJitLink de CUDA 12.5 de TensorFlow y el NVVM de Numba). Solo se midieron las referencias: NumPy (CPU) 70.7 ms y cuBLAS vía `tf.matmul` 2.77 ms (≈ 26×).\n\n" + PEND)

if cvdf is not None:
    c = cvdf.copy()
    t = pd.DataFrame({"Configuración": c.config, "Épocas (mejor)": c.epocas_ejecutadas.astype(str) + " (" + c.mejor_epoca.astype(str) + ")",
                      "fit (s)": c.fit_s.map(lambda x: f(x, 1)), "Época estable (s)": c.epoca_estable_s.map(lambda x: f(x, 2)),
                      "Acc. val. máx.": (100 * c.val_acc_mejor).map(lambda x: f(x, 1) + " %"), "Acc. test": (100 * c.test_acc).map(lambda x: f(x, 1) + " %"),
                      "CPU estimado (s)": c.cpu_estimado_s.map(lambda x: f(x, 0)), "Speedup est.": c.speedup_estimado_vs_CPU.map(lambda x: f(x, 1) + "×")})
    sec_cv = (tbl(t) + "\n\n*«CPU estimado» = épocas ejecutadas × " + f(cpu['epoca_estable_s'], 1) + " s (época estable medida en CPU); es una extrapolación, no una medición.*\n\n"
              "![Convergencia](../results/complemento_convergencia.png)\n\n" +
              f"**Lectura.** Entrenando hasta converger, el GPU alcanza **{f(100*c.test_acc.min(),1)}–{f(100*c.test_acc.max(),1)} % de exactitud en test**, en línea con el 95.7 % que obtuvo la CNN 2 en el Lab 3: el GPU acelera el cálculo sin cambiar lo que aprende la red. Con el protocolo del Lab 3 (bs=32) el entrenamiento terminó en {int(c.epocas_ejecutadas.iloc[0])} épocas ({f(c.fit_s.iloc[0],0)} s en GPU frente a ≈ {f(c.cpu_estimado_s.iloc[0]/60,0)} min estimados en CPU, {f(c.speedup_estimado_vs_CPU.iloc[0],0)}×); `EarlyStopping` lo detuvo temprano por las oscilaciones de la pérdida de validación que provoca BatchNorm (visibles en la gráfica). **Con 4 épocas la exactitud de la corrida principal (≈ 69-72 %) era simplemente sub-entrenamiento.** Con batch 128 la red llega a {f(100*c.test_acc.iloc[1],1)} % (fp32) y {f(100*c.test_acc.iloc[2],1)} % (fp16): la precisión mixta casi no cambia el resultado ({f(100*(c.test_acc.iloc[1]-c.test_acc.iloc[2]),1)} puntos menos) y acelera la época de {f(c.epoca_estable_s.iloc[1],2)} s a {f(c.epoca_estable_s.iloc[2],2)} s. La 4.ª fila (30 épocas máx.) se detuvo también en la época 15 con los mismos pesos que la 3.ª (misma semilla), por lo que es redundante en exactitud; sus tiempos distintos ({f(c.fit_s.iloc[3],1)} s vs. {f(c.fit_s.iloc[2],1)} s para el mismo entrenamiento) muestran la variabilidad de tiempo de Colab entre ejecuciones idénticas (≈ {f(100*(c.fit_s.iloc[2]-c.fit_s.iloc[3])/c.fit_s.iloc[2],0)} %).")
else:
    sec_cv = PEND

resumen_conv = (f"- **Entrenando hasta converger** (protocolo del Lab 3) el GPU alcanza {f(100*cvdf.test_acc.min(),1)}–{f(100*cvdf.test_acc.max(),1)} % de exactitud en test (Lab 3: 95.7 %) en {f(cvdf.fit_s.min(),0)}–{f(cvdf.fit_s.max(),0)} s; en CPU el mismo entrenamiento tomaría ≈ {f(cvdf.cpu_estimado_s.iloc[0]/60,0)}–{f(cvdf.cpu_estimado_s.max()/60,0)} min (estimado)." if cvdf is not None else "")
# ---------------- documento ----------------
md = f"""---
title: "Proyecto 3 — Mejorando algoritmos con CUDA: Informe de rendimiento y análisis"
---

# Proyecto 3 — Mejorando algoritmos con CUDA
**Informe de Rendimiento y Análisis** · CC3069 Computación Paralela y Distribuida · UVG · Semestre II 2026

Integrantes: Roberto Nájera, Genser Catalán, Estuardo Castro · Repositorio: <https://github.com/Ultimate-Truth-Seeker/Proyecto3CPyD>

## Resumen
Se aceleró el clasificador ASL del Laboratorio 3 de Data Science (CNN 2, 29 clases, imágenes 64×64×3, 20 300 imágenes) separando el trabajo en dos recursos: **la preparación de imágenes (leer JPG, redimensionar, normalizar) se reparte entre los núcleos del CPU**, y **el entrenamiento y la inferencia de la red se ejecutan en el GPU con CUDA**. Medido en Google Colab (Tesla T4, 2 vCPU):

- **Entrenamiento (4 épocas): {f(cpu['fit_s'],0)} s → {f(g32['fit_s'],1)} s, speedup {sp(cpu['fit_s'], g32['fit_s'])}** (época en régimen estable: {f(cpu['epoca_estable_s'],0)} s → {f(g32['epoca_estable_s'],1)} s, {sp(cpu['epoca_estable_s'], g32['epoca_estable_s'])}).
- **Preparación de imágenes: {f(ps,1)} s → {f(pp,1)} s, speedup {sp(ps, pp)}** — modesto porque el Colab gratuito expone 2 vCPU que son **un solo núcleo físico con 2 hilos**.
- **Extremo a extremo (preparar + entrenar + inferir): {f(e2e_seq,0)} s → {f(e2e32,0)} s, speedup {sp(e2e_seq, e2e32)}.** Tras acelerar el entrenamiento, la preparación en CPU pasa de ser el {f(100*ps/e2e_seq,0)} % del tiempo a ser el {f(100*pp/e2e32,0)} %: el cuello de botella se desplaza.
{resumen_conv}
- La utilización del GPU fue baja (media {f(u['B_entrenamiento_GPU']['gpu'],0)} %): con una red pequeña y batch de 32, el GPU queda subutilizado.

## 1. Entorno y metodología

| Elemento | Valor |
|---|---|
| GPU | NVIDIA Tesla T4, 15 360 MiB, driver 580.82.07 (CUDA 13.0); TensorFlow compilado con CUDA 12.5.1 y cuDNN 9 |
| CPU | Intel Xeon @ 2.00 GHz, **2 vCPU = 1 núcleo físico × 2 hilos** (`lscpu`), 13.6 GB RAM |
| Software | Python 3.13, TensorFlow 2.21.0, Keras 3.13.2 |
| Datos | ASL Alphabet (Kaggle); 700 imágenes/clase → 20 300; split estratificado 70/15/15 = 14 210 / 3 045 / 3 045; semilla 42 |
| Modelo | CNN 2 del Lab 3 (3×[Conv+BatchNorm+MaxPool], Dense 256, Dense 128, softmax 29), Adam, lr 5·10⁻⁴ |
| Condiciones | **4 épocas fijas** (sin `EarlyStopping`) para que CPU y GPU hagan exactamente el mismo trabajo; batch 32; 3 repeticiones; warm-up sin medir; mismos datos y semilla |

**Procedimiento.** (1) El manifiesto y el split se construyen una sola vez, de modo que secuencial y paralelo procesan las mismas 20 300 rutas. (2) Se hace una *lectura en frío* de todos los archivos para dejarlos en el caché del sistema operativo; así las repeticiones comparan cómputo y no el disco de Colab. (3) Cada fase se repite 3 veces y se reporta promedio ± desviación estándar. (4) La versión paralela de la preparación se verifica **idéntica bit a bit** a la secuencial (`np.array_equal`). (5) En GPU, los arreglos NumPy se pasan directo a `fit()`, por lo que el tiempo **incluye las transferencias host→device** de cada batch. (6) Un hilo en segundo plano registra cada 1 s el uso de cada vCPU, la utilización y memoria del GPU (`nvidia-smi`) y la RAM; su costo es el mismo en todas las fases.

## 2. Análisis de la solución secuencial (Paso 1)
La tubería del Lab 3 es: *descarga/validación del dataset → manifiesto con submuestra → split estratificado → leer, decodificar, redimensionar y normalizar cada imagen → construir tensores → entrenar → evaluar*. Las tablas completas de operaciones (Tabla 1) y de operaciones repetitivas (Tabla 2) están en la sección 1 del notebook `proyecto3_cuda.ipynb`. Los dos bucles pesados son:

| Operación repetitiva | Datos que procesa | Repeticiones |
|---|---|---|
| Leer + decodificar + redimensionar + normalizar | 1 imagen JPG 200×200×3 (independiente de las demás) | 20 300 |
| Paso de entrenamiento (adelante, pérdida, atrás, Adam) | 1 mini-batch de 32 imágenes 64×64×3 | 445 por época × 4 épocas |
| Evaluación de validación / test | lotes de 32 imágenes | 96 lotes por época / 96 lotes |

## 3. Paralelización y diseño (Pasos 3 y 4)
| Operación | Dónde | Justificación |
|---|---|---|
| Leer, decodificar, redimensionar y normalizar cada imagen | **CPU, procesos** (`multiprocessing`, contexto `spawn`, `chunksize=64`) | 20 300 tareas independientes; la decodificación JPEG (Huffman) es un flujo de bits con ramas y acceso a archivos: no encaja en el modelo SIMT del GPU. |
| Convoluciones, BatchNorm, pooling, multiplicaciones matriciales, retropropagación, Adam | **GPU (CUDA: cuDNN y cuBLAS)** | Operaciones regulares, densas, de alta intensidad aritmética, con millones de elementos independientes. |
| Bucle de mini-batches y épocas | secuencial | El paso *k+1* necesita los pesos del paso *k* (límite de Amdahl). |

**Mapeo CUDA.** Un hilo calcula un elemento del tensor de salida; un bloque, un *tile* del tensor de salida; el grid cubre `batch × alto × ancho × filtros`. Las 32 imágenes del batch se procesan a la vez (primera dimensión del grid). Se implementó además un kernel propio para `Dense 4608→256 + ReLU` con bloque de 16×16 hilos y grid `⌈256/16⌉ × ⌈M/16⌉` (sección 5.4).

## 4. Resultados de rendimiento (Pasos 2, 5 y 6)

### 4.1 Preparación de imágenes (Fase A), en segundos
{tbl(t_prep)}

Escalamiento con el número de procesos (1 repetición por punto):

{tbl(t_sw)}

![Escalamiento de la fase A](../results/prep_escalamiento.png)

El speedup se estanca en ≈ 1.2× porque las 2 vCPU de Colab son **un solo núcleo físico con hyper-threading**: dos procesos comparten las mismas unidades de ejecución y, con 4 u 8 procesos, solo se añade cambio de contexto. La eficiencia con 2 procesos es {f((ps/pp)/N_CPU,2)}. En una máquina con más núcleos físicos se esperaría un escalamiento mucho mayor.

### 4.2 Entrenamiento e inferencia (Fases B y C)
Resumen (promedio de 3 ejecuciones):

{tbl(t_sum)}

Detalle por ejecución:

{tbl(t_runs)}

### 4.3 Tabla comparativa y speedup por fase
{tbl(t_speed)}

![Comparación de tiempos y speedup](../results/comparacion_tiempos_speedup.png)

Notas para interpretar la tabla:

- **La 1.ª época en GPU cuesta ≈ {f(g32['primera_epoca_s'],0)} s frente a {f(g32['epoca_estable_s'],1)} s de las siguientes**: inicialización de cuDNN, selección de algoritmos y trazado del grafo. Con solo 4 épocas ese costo fijo pesa mucho (por eso B′ muestra {sp(cpu['epoca_estable_s'], g32['epoca_estable_s'])} y B {sp(cpu['fit_s'], g32['fit_s'])}); con más épocas el speedup global se acerca al de la época estable.
- **La inferencia mejora poco ({sp(cpu['inferencia_s'], g32['inferencia_s'])}).** Son solo 3 045 imágenes y el tiempo está dominado por costos fijos de `predict()` (construcción del grafo, conversión de datos), no por cómputo; además tiene alta variabilidad (σ = {f(g32['inferencia_std'])} s).
- La comparación bs=128+fp16 **cambia también el tamaño de batch**; su speedup de época ({sp(cpu['epoca_estable_s'], g128['epoca_estable_s'])}) mezcla el efecto del GPU con el de procesar 4× más imágenes por paso.
- **Límite de Amdahl.** En la línea base la preparación es el {f(100*ps/e2e_seq,1)} % del tiempo. Si el entrenamiento y la inferencia fueran gratis, el speedup máximo sin acelerar la fase A sería {f(e2e_seq/ps,0)}×; el observado ({sp(e2e_seq, e2e32)}) queda por debajo por la inicialización del GPU y la fase A.
- **Variabilidad.** La primera ejecución en GPU fue más lenta ({f(runs[(runs.config.str.startswith('GPU (CUDA'))&(runs.run==1)].train_s.iloc[0],1)} s vs. ≈ 28 s); Colab es una máquina compartida y la T4 operaba a 74 °C.

## 5. Evidencia de ejecución y análisis

### 5.1 Uso de recursos medido
{tbl(t_use)}

| Fase | Gráfica |
|---|---|
| Preparación secuencial | ![](../results/uso_A_prep_secuencial.png) |
| Preparación paralela | ![](../results/uso_A_prep_paralelo_CPU.png) |
| Entrenamiento en CPU | ![](../results/uso_B_entrenamiento_CPU.png) |
| Entrenamiento en GPU (bs=32) | ![](../results/uso_B_entrenamiento_GPU.png) |
| Entrenamiento en GPU (bs=128, fp16) | ![](../results/uso_B_entrenamiento_GPU_optimizado.png) |

Lectura: en la preparación **secuencial** el CPU promedia {f(u['A_prep_secuencial']['cpu'],0)} % (un solo proceso Python); en la **paralela** ambas vCPU están al {f(u['A_prep_paralelo_CPU']['cpu'],0)} %. En el **entrenamiento en CPU** ambas vCPU trabajan al {f(u['B_entrenamiento_CPU']['cpu'],0)} % y el GPU permanece en 0 %. En el **entrenamiento en GPU** el entorno reconoce y usa la T4: se asignan ≈ {f(u['B_entrenamiento_GPU']['mem'],0)} MB de memoria de GPU y la utilización llega a {f(u['B_entrenamiento_GPU']['gpu_max'],0)} % con una media de solo {f(u['B_entrenamiento_GPU']['gpu'],0)} %.

**El GPU no está saturado.** La gráfica muestra un patrón de picos (≈ 60 %) intercalados con valles en 0 %, que tienden a coincidir con picos del CPU al 100 %. Es consistente con que el CPU prepara cada batch desde NumPy, lanza los kernels y ejecuta la validación mientras el GPU espera (no se aisló cada causa con un perfilador). Con un modelo pequeño y batches de 32 imágenes de 64×64, cada kernel dura microsegundos y el costo de lanzamiento y de Python/TensorFlow por paso es comparable al cómputo. En la variante bs=128+fp16 la época baja a {f(g128['epoca_estable_s'],2)} s (≈ {f(g32['epoca_estable_s']/g128['epoca_estable_s'],1)}× más rápida que bs=32) pero la utilización media del GPU **baja** a {f(u['B_entrenamiento_GPU_optimizado']['gpu'],0)} %: el cuello de botella ya no es el cómputo sino la alimentación de datos y el lanzamiento de kernels.

*Nota sobre los logs de `nvidia-smi`:* las capturas `results/nvidia_smi_durante_entrenamiento_*.txt` se tomaron al inicio de la 2.ª época, instante en que el GPU estaba entre épocas (0 % de utilización, pero 2.2 GB asignados y el proceso `python3` visible en la tabla de procesos). La utilización real durante el entrenamiento está en las series de `results/uso_B_entrenamiento_GPU*.csv`.

### 5.1b Capturas de pantalla del entorno (Colab)
Capturas del cuaderno junto al panel *Recursos* de Colab durante la ejecución:

| Captura | Qué muestra |
|---|---|
| ![](../results/capturas/colab_01_entrenamiento_CPU.png) | Entrenamiento en **CPU** (celda 9, 35 min): la gráfica del monitor muestra ambas vCPU saturadas y GPU en 0 %; el panel marca **RAM de GPU 0.1 / 15.0 GB** (sin uso). |
| ![](../results/capturas/colab_02_entrenamiento_GPU_optimizado.png) | Entrenamiento en **GPU** (celda 13): `nvidia-smi` lista el proceso `python3` con ≈ 2210 MiB y el panel marca **RAM de GPU 2.2 / 15.0 GB**. |
| ![](../results/capturas/colab_03_complemento_convergencia.png) | Notebook complementario al terminar el entrenamiento hasta convergencia (curvas de validación de las 4 configuraciones) con **RAM de GPU 1.8 / 15.0 GB**. |

### 5.2 Equivalencia de resultados y convergencia
Con 4 épocas fijas la exactitud en test es: CPU {f(100*cpu['acc_test'],1)} %, GPU bs=32 {f(100*g32['acc_test'],1)} % (rango {f(100*runs[runs.config.str.startswith('GPU (CUDA')].test_acc.min(),1)}–{f(100*runs[runs.config.str.startswith('GPU (CUDA')].test_acc.max(),1)} %), GPU bs=128+fp16 {f(100*g128['acc_test'],1)} %. **Estas cifras no representan la calidad final del modelo**: el Lab 3 entrenó hasta 15 épocas con `EarlyStopping` y obtuvo 95.7 %; aquí el entrenamiento se corta a propósito a las 4 épocas para medir tiempo con trabajo idéntico, y con BatchNorm la exactitud oscila bastante entre épocas. La diferencia CPU/GPU está dentro de esa variabilidad (el CPU es determinista; el GPU varía entre ejecuciones por el no determinismo de los kernels y el orden de las operaciones de punto flotante). La variante bs=128+fp16 hace 4× menos actualizaciones por época (112 vs. 445) con el mismo learning rate, por lo que a las 4 épocas está claramente sub-entrenada: **sus tiempos no son comparables en calidad**.

Entrenamiento hasta convergencia en GPU (protocolo del Lab 3: máx. 15 épocas, `EarlyStopping`):

{sec_cv}

### 5.3 Costo de la transferencia CPU ↔ GPU
{sec_tr}

### 5.4 Kernel CUDA propio (`Dense 4608→256 + ReLU`, M = 3 045)
{sec_k}

## 6. Discusión técnica

### 6.1 ¿Por qué es más eficiente procesar las imágenes/archivos con los múltiples núcleos del CPU y delegar exclusivamente la matemática de la red neuronal al GPU?
- **Naturaleza del trabajo.** Decodificar un JPEG es recorrer un flujo de bits con decodificación de Huffman (cada símbolo depende del anterior), con muchas ramas y acceso a archivos. Los hilos de un *warp* ejecutan la misma instrucción a la vez; si cada imagen sigue un camino distinto, se serializan (divergencia). El CPU, con pocos núcleos muy flexibles (predicción de ramas, cachés grandes), maneja bien este patrón, y como cada imagen es independiente se reparte entre procesos sin comunicación.
- **Entrada/salida.** Los archivos viven en el disco/RAM del *host*; el GPU no los lee directamente. Decodificar en GPU (nvJPEG/DALI) es posible, pero con imágenes de 200×200 añade complejidad y el beneficio es pequeño.
- **La matemática de la red es regular:** las mismas operaciones de multiplicar-acumular sobre millones de elementos independientes, sin ramas y con alta reutilización de datos. Es lo que el GPU hace mejor (miles de hilos, memoria compartida, Tensor Cores).
- **Transferencias.** PCIe es mucho más lento que la memoria del GPU. Se preprocesa **una sola vez** en el CPU y solo se mueven tensores compactos y listos para usar.
- **Dónde está el tiempo.** La preparación ocurre una vez ({f(ps,0)} s en secuencial, {f(100*ps/e2e_seq,0)} % del total), mientras que el entrenamiento repite 445 pasos por época ({f(cpu['fit_s'],0)} s en CPU). Delegar la red al GPU recupera casi todo el tiempo; acelerar además la preparación solo importa cuando el entrenamiento ya es rápido (aquí lo es: pasa a ser el {f(100*pp/e2e32,0)} % del total).

### 6.2 ¿Cómo CUDA permitió reducir o cambiar el tiempo de procesamiento?
- Una época pasó de **{f(cpu['epoca_estable_s'],0)} s a {f(g32['epoca_estable_s'],1)} s ({sp(cpu['epoca_estable_s'], g32['epoca_estable_s'])})**: miles de hilos calculan en paralelo los elementos de salida de cada convolución y de cada multiplicación matricial (cuDNN/cuBLAS), mientras el CPU, con un solo núcleo físico, ejecuta esas mismas operaciones con SIMD y poca concurrencia. Entrenar 4 épocas pasó de {f(cpu['fit_s']/60,1)} min a {f(g32['fit_s'],0)} s ({sp(cpu['fit_s'], g32['fit_s'])}); el costo fijo de inicialización del GPU (≈ {f(g32['primera_epoca_s']-g32['epoca_estable_s'],0)} s en la 1.ª época) se amortiza con más épocas.
- Con el protocolo completo del Lab 3 (hasta converger) el entrenamiento en GPU tarda ≈ {f(cvdf.fit_s.iloc[0],0)} s frente a ≈ {f(cvdf.cpu_estimado_s.iloc[0]/60,0)} min estimados en CPU (≈ {f(cvdf.speedup_estimado_vs_CPU.iloc[0],0)}×), con exactitud equivalente.
- **Cambió también el perfil del tiempo:** de {f(100*cpu['fit_s']/e2e_seq,0)} % entrenamiento + {f(100*ps/e2e_seq,0)} % preparación a una distribución más repartida (preparación {f(100*pp/e2e32,0)} %, entrenamiento {f(100*g32['fit_s']/e2e32,0)} %, inferencia {f(100*g32['inferencia_s']/e2e32,0)} %).
- **Ajustes dentro del GPU.** Aumentar el batch a 128 y usar precisión mixta (Tensor Cores) redujo la época a {f(g128['epoca_estable_s'],2)} s, pero con menor utilización de GPU, lo que confirma que con este modelo pequeño el límite pasó a ser la alimentación de datos y el lanzamiento de kernels, no la capacidad aritmética del GPU.

### 6.3 ¿Qué factores justifican o limitan la utilización de CUDA?
**Justifican:** operaciones densas y regulares; alto paralelismo de datos (batch × píxeles × filtros); alta intensidad aritmética; una carga que se repite cientos de veces (pasos × épocas), lo que amortiza la inicialización y las transferencias.

**Limitan (observados en este experimento):**
1. **Amdahl:** el bucle de pasos/épocas es secuencial y la preparación en CPU no se acelera con el GPU; con el entrenamiento ya rápido, la fase A es el {f(100*pp/e2e32,0)} % del total.
2. **Tamaño del problema:** batch de 32 imágenes de 64×64 y una red de pocas capas no saturan una T4 (utilización media {f(u['B_entrenamiento_GPU']['gpu'],0)} %). Con modelos o batches mayores el speedup crecería.
3. **Costos fijos:** inicialización de cuDNN y trazado del grafo (≈ {f(g32['primera_epoca_s']-g32['epoca_estable_s'],0)} s), y *overhead* de lanzamiento de kernels y de Python/TensorFlow por paso. La inferencia de 3 045 imágenes solo mejora {sp(cpu['inferencia_s'], g32['inferencia_s'])} por este motivo.
4. **Transferencias PCIe** y memoria del GPU (15 GB en la T4; el entrenamiento usó ≈ 2.2 GB).
5. **Recursos del CPU de Colab:** 1 núcleo físico limita la fase A (speedup {sp(ps, pp)}) y alimenta al GPU con lentitud.
6. **Portabilidad/ecosistema:** versiones de CUDA, cuDNN y nvJitLink deben ser compatibles; en este proyecto Numba falló por esa causa (ver 5.4).

### 6.4 Funciones y librerías adicionales que usan el GPU
CUDA runtime y driver; **cuDNN** (convoluciones, BatchNorm, pooling, selección de algoritmos); **cuBLAS** (GEMM de las capas densas); **TensorFlow/Keras** (`tf.device`, despacho de kernels, `keras.mixed_precision` para FP16 en Tensor Cores); **Numba** (`numba.cuda`: `cuda.jit`, `cuda.shared.array`, `cuda.syncthreads`) y **CuPy** (`RawKernel`, NVRTC) para los kernels propios; **`nvidia-smi`** para monitoreo.

## 7. Limitaciones del estudio
- Un único entorno (Colab gratuito, 1 núcleo físico); con un CPU más potente, el speedup frente al CPU sería menor, y el de la fase A, mayor.
- 4 épocas fijas para medir tiempo, no calidad; la comparación de exactitud se complementa con el entrenamiento hasta convergencia (5.2).
- Speedups de época en régimen estable excluyen la 1.ª época; el speedup de `fit` la incluye.
- El monitor de recursos agrega un costo pequeño y común a todas las fases.

## 8. Conclusiones
1. Separar el trabajo funcionó: **CPU para archivos, GPU para la matemática de la red**. El entrenamiento se aceleró {sp(cpu['fit_s'], g32['fit_s'])} (4 épocas) y la época en régimen estable {sp(cpu['epoca_estable_s'], g32['epoca_estable_s'])}; el proceso completo, {sp(e2e_seq, e2e32)}.
2. La preparación en CPU solo ganó {sp(ps, pp)} con 2 procesos porque Colab ofrece un único núcleo físico; el diseño escala con los núcleos disponibles.
3. El GPU quedó subutilizado (≈ {f(u['B_entrenamiento_GPU']['gpu'],0)} %): el límite está en el tamaño del problema y la alimentación de datos, no en la capacidad del hardware. Aumentar el batch o la complejidad del modelo es el siguiente paso natural.
4. El GPU no compromete la calidad: entrenado hasta converger llega a {f(100*cvdf.test_acc.min(),1)}–{f(100*cvdf.test_acc.max(),1)} % en test (Lab 3: 95.7 %). El tiempo de transferencia es pequeño con memoria *pinned* ({f(trdf[trdf.metodo.str.contains('pinned')].GBps.iloc[0],1)} GB/s) y mayor con memoria paginada.
5. La ganancia depende de amortizar costos fijos (inicialización, transferencias, preparación): conviene en entrenamientos con muchas épocas y no en inferencias cortas.

## 9. Contribuciones
| Integrante | Contribución |
|---|---|
| Roberto Nájera | Análisis del código secuencial, diseño e implementación de la versión paralela (CPU multi-proceso y GPU/CUDA), mediciones en Colab, análisis e informe. |
| Genser Catalán | _(completar)_ |
| Estuardo Castro | _(completar)_ |

## 10. Reproducibilidad
`notebooks/proyecto3_cuda.ipynb` (experimento principal, ejecutado, con salidas) y `notebooks/proyecto3_complemento.ipynb` (kernels, transferencias y convergencia). Datos crudos de las mediciones en `results/`. Este informe se genera con `python docs/build_informe.py .` a partir de esos CSV.
"""
out = repo / "docs" / "informe_rendimiento.md"; out.write_text(md)
print("md escrito:", out, len(md), "caracteres")

if "--pdf" in sys.argv:
    import markdown
    body = markdown.markdown(md.split("---\n", 2)[2], extensions=["tables", "sane_lists"])
    css = """body{font-family:'DejaVu Sans',Arial,sans-serif;font-size:10.5px;line-height:1.45;margin:0 8px;color:#111}
    h1{font-size:21px;border-bottom:2px solid #1f4e79;padding-bottom:4px;color:#1f4e79} h2{font-size:15px;color:#1f4e79;margin-top:20px;border-bottom:1px solid #ccc}
    h3{font-size:12.5px;color:#333} table{border-collapse:collapse;margin:8px 0;width:100%;font-size:9.5px;page-break-inside:avoid}
    th,td{border:1px solid #bbb;padding:3px 5px;text-align:left;vertical-align:top} th{background:#e8eef5} img{max-width:100%;margin:4px 0}
    blockquote{background:#fff7e0;border-left:4px solid #e0a800;margin:8px 0;padding:4px 10px} code{background:#f2f2f2;padding:0 2px;font-size:9.5px}"""
    html = f"<html><head><meta charset='utf-8'><style>{css}</style></head><body>{body}</body></html>"
    h = repo / "docs" / "_informe.html"; h.write_text(html)
    pdf = repo / "docs" / "informe_rendimiento.pdf"
    subprocess.run(["wkhtmltopdf", "--quiet", "--enable-local-file-access", "-s", "Letter", "-T", "14mm", "-B", "14mm", "-L", "12mm", "-R", "12mm", str(h), str(pdf)], check=True)
    h.unlink(); print("pdf:", pdf)
