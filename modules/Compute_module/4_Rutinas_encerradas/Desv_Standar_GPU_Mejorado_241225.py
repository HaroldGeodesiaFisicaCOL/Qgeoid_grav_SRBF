#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import sys
import math
import numpy as np
import pandas as pd
from datetime import datetime
from rasterio.transform import from_origin
from rasterio.crs import CRS
import rasterio
from multiprocessing import Pool, cpu_count
import multiprocessing
import functools
import gc

# ------------------ helper: raster + gravedad (copiado y adaptado) ------------------
# constantes (usa las tuyas)
UO = 62636860.850
WO = 62636853.4
gamma_a = 978032.67715
gamma_b = 983218.63685
c1 = 0.0053024
c2 = 0.0000058
f = 0.00335281068118
m_par = 0.00344978600308
# Elipsoide GRS80
a = 6378137.0
b = 6356752.3141
e2 = 0.00669438002290

def gravedad_teorica_somigliana(lat):
    lat_r = np.radians(lat)
    num = (a * gamma_a * np.cos(lat_r)**2 + b * gamma_b * np.sin(lat_r)**2)
    den = np.sqrt(a**2 * np.cos(lat_r)**2 + b**2 * np.sin(lat_r)**2)
    return num / den

def gamma_h(lat, gamma_0, h):
    lat_r = np.radians(lat)
    return gamma_0 * (1 - (2/a)*(1 + f + m_par - 2*f*np.sin(lat_r)**2)*h + (3/(a**2))*(h**2))

def calculo_desv_modelo(df):
    df['altura_norm'] = np.where(df['SRTM'] == 0,
                                  -df['anom_res'],
                                  df['altura'] - df['anom_res'])
    lat = df['Latitud'].values
    gamma0 = gravedad_teorica_somigliana(lat)
    df['grav_teo'] = gamma0
    Hn = df['altura_norm'].values
    df['grav_telu'] = gamma_h(lat, gamma0, Hn)
    df['Std_Zeta'] = df['Std_T'] / (df['grav_telu'] / 1e5)
    return df

def generar_raster(df, output_filename, pixel_size):
    latitudes = np.sort(df['Latitud'].unique())[::-1]
    longitudes = np.sort(df['Longitud'].unique())
    h = len(latitudes)
    w = len(longitudes)
    raster = np.full((h, w), np.nan, dtype=np.float32)
    lat_idx = {lat: i for i, lat in enumerate(latitudes)}
    lon_idx = {lon: j for j, lon in enumerate(longitudes)}
    for _, row in df.iterrows():
        i = lat_idx[row['Latitud']]
        j = lon_idx[row['Longitud']]
        raster[i, j] = row['Std_Zeta']
    transform = from_origin(longitudes[0] - pixel_size/2,
                            latitudes[0] + pixel_size/2,
                            pixel_size, pixel_size)
    crs = CRS.from_wkt(
        'GEOGCS["GRS80",DATUM["Geodetic_Reference_System_1980",'
        'SPHEROID["GRS 1980",6378137,298.257222101]],'
        'PRIMEM["Greenwich",0],UNIT["degree",0.017453292519933]]'
    )
    with rasterio.open(
        output_filename, 'w', driver='GTiff',
        height=h, width=w, count=1, dtype=raster.dtype,
        crs=crs, transform=transform, nodata=np.nan
    ) as dst:
        dst.write(raster, 1)
    print(f"Ráster generado: {output_filename} {datetime.now().strftime('%H:%M:%S')}", flush=True)

# ------------------ globales del worker (serán inicializados por init_worker) ------------------
M_memmap = None
B_memmap = None
M_shape = None
B_shape = None
TILE_M = None

def init_worker(path_M, path_B, tile_m):
    """Initializer for worker processes: carga memmaps (lectura) en cada proceso."""
    global M_memmap, B_memmap, M_shape, B_shape, TILE_M
    # Carga memmaps de solo lectura; evita copia completa en memoria del proceso padre
    M_memmap = np.load(path_M, mmap_mode='r')
    B_memmap = np.load(path_B, mmap_mode='r')
    M_shape = M_memmap.shape
    B_shape = B_memmap.shape
    TILE_M = int(tile_m)
    # hint para BLAS threads (estos variables deben estar definidos antes de fork para muchas BLAS)
    # No configuramos aquí threads; se recomienda pasarlo por OMP_NUM_THREADS / MKL_NUM_THREADS al lanzar
    # Forzar GC un poco
    gc.collect()

def process_batch(args):
    """Procesa un batch de filas [i0:i1) de B_memmap.
    Retorna (i0, i1, stds_array)
    """
    i0, i1 = args
    global M_memmap, B_memmap, M_shape, TILE_M
    r = i1 - i0
    m = M_shape[0]
    # Leer B_batch como copia contigua (float64)
    B_batch = np.asarray(B_memmap[i0:i1, :], dtype=np.float64, order='C')
    acc = np.zeros((r,), dtype=np.float64)

    # Tile por columnas de M
    for j0 in range(0, m, TILE_M):
        j1 = min(j0 + TILE_M, m)
        # leer bloque de M: M[:, j0:j1] -> shape (m, t)
        M_tile = np.asarray(M_memmap[:, j0:j1], dtype=np.float64, order='F')  # column-major may help -> but keep as contiguous
        # tmp = B_batch @ M_tile  -> (r x t)
        # nota: M_tile es (m x t), B_batch (r x m)
        tmp = B_batch.dot(M_tile)  # BLAS accelerated
        # b_tile = B_batch[:, j0:j1]
        b_tile = B_batch[:, j0:j1]
        # acumular suma de elemento-por-elemento por fila
        # equivalente a acc += np.sum(tmp * b_tile, axis=1)
        acc += np.einsum('ij,ij->i', tmp, b_tile)
        # liberar memoria temporal
        del M_tile, tmp, b_tile
        gc.collect()

    # sqrt de la suma acumulada (valores deben ser >= 0 numéricamente; forzar no-negativos)
    acc = np.where(acc < 0, 0.0, acc)
    stds = np.sqrt(acc, dtype=np.float64)
    return (i0, i1, stds)

def calc_std_cpu_matrix_free(path_var_cor_npy, path_pixeles, path_B, output_raster,
                             pixel_size, batch_size=500, tile_m=512, num_workers=1,
                             blas_threads=None, save_pixels=True):
    """
    path_var_cor_npy: .npy para M (m x m)
    path_B: .npy para B (n x m)
    path_pixeles: archivo .tsv con puntos (se reescribe con Std_T)
    batch_size: filas de B por trabajo
    tile_m: número de columnas de M por tile (ajustar según RAM)
    num_workers: procesos paralelos (1 = secuencial)
    blas_threads: si no None, fuerza OMP/MKL threads (antes de fork)
    """
    if blas_threads is not None:
        os.environ['OMP_NUM_THREADS'] = str(blas_threads)
        os.environ['OPENBLAS_NUM_THREADS'] = str(blas_threads)
        os.environ['MKL_NUM_THREADS'] = str(blas_threads)
        os.environ['VECLIB_MAXIMUM_THREADS'] = str(blas_threads)

    # Abrir memmaps en el proceso principal para comprobar shapes
    print("[1/6] Abriendo memmaps (solo lectura)...", flush=True)
    M_mm = np.load(path_var_cor_npy, mmap_mode='r')
    B_mm = np.load(path_B, mmap_mode='r')
    if M_mm.shape[0] != M_mm.shape[1]:
        raise ValueError("M debe ser cuadrada")
    if M_mm.shape[0] != B_mm.shape[1]:
        raise ValueError(f"Mismatch shapes: M: {M_mm.shape}, B: {B_mm.shape}")
    m = M_mm.shape[0]
    n = B_mm.shape[0]
    print(f"  M: {M_mm.shape}  (≈{M_mm.nbytes/1e9:.2f} GB), B: {B_mm.shape} (≈{B_mm.nbytes/1e9:.2f} GB)", flush=True)

    # Crear lista de batches
    indices = []
    for i0 in range(0, n, batch_size):
        i1 = min(i0 + batch_size, n)
        indices.append((i0, i1))

    std_T = np.empty(n, dtype=np.float64)

    print(f"[2/6] Preparando pool (workers={num_workers}) y tile_m={tile_m} batch_size={batch_size}", flush=True)
    # Launch pool
    pool = None
    try:
        if num_workers == 1:
            # secuencial: inicializar worker globals en main thread
            init_worker(path_var_cor_npy, path_B, tile_m)
            for (i0, i1) in indices:
                print(f"  Procesando filas {i0}:{i1} ...", flush=True)
                _, _, stds = process_batch((i0, i1))
                std_T[i0:i1] = stds
        else:
            # multiprocessing pool
            # init_worker será llamado en cada proceso con los mismos memmaps
            pool = Pool(processes=num_workers, initializer=init_worker,
                        initargs=(path_var_cor_npy, path_B, tile_m))
            # map (se puede usar imap_unordered para empezar a recibir resultados antes)
            for res in pool.imap_unordered(process_batch, indices):
                i0, i1, stds = res
                std_T[i0:i1] = stds
                print(f"  Batch {i0}:{i1} completado (colocado en resultado).", flush=True)
    finally:
        if pool:
            pool.close()
            pool.join()

    # Post-procesado: guardar en archivo de pixeles y raster
    print("[3/6] Post-procesando DataFrame y raster...", flush=True)
    df = pd.read_csv(path_pixeles, sep='\t', header=0)
    if len(df) != n:
        print(f"[WARN] número de puntos en {path_pixeles} ({len(df)}) != filas en B ({n}). Se intentará ajustar por índice mínimo.", flush=True)
        # trunca o extiende según corresponda
        minlen = min(len(df), n)
        df = df.iloc[:minlen].copy()
        std_T = std_T[:minlen]

    df["Std_T"] = std_T.astype(np.float32)
    df = calculo_desv_modelo(df)
    generar_raster(df, output_raster, pixel_size / 60.0)
    if save_pixels:
        df.to_csv(path_pixeles, sep="\t", index=False)
    print("[OK] Proceso completado correctamente", flush=True)


# ------------------ CLI ------------------
def parse_args():
    p = argparse.ArgumentParser(description="Std residual CPU (matrix-free, memmap, tile+batch)")
    p.add_argument('path_var_cor', help=".npy matrix M (m x m)")
    p.add_argument('path_pixeles', help="archivo .tsv con puntos (tiene columnas Latitud, Longitud, Std_T etc.)")
    p.add_argument('path_B', help=".npy matrix B (n x m)")
    p.add_argument('output_raster', help="ruta salida GeoTIFF")
    p.add_argument('pixel_size', type=float, help="pixel size (en minutos en tu pipeline original) — aquí se divide por 60 internamente")
    p.add_argument('--batch_size', type=int, default=500, help='Filas de B por batch (memoria por batch ≈ batch_size * m * 8 bytes)')
    p.add_argument('--tile_m', type=int, default=512, help='Columnas de M por tile (memoria ≈ m * tile_m * 8 bytes)')
    p.add_argument('--num_workers', type=int, default=1, help='Procesos paralelos (I/O-bound -> usa con cuidado)')
    p.add_argument('--blas_threads', type=int, default=None, help='Forzar OMP/MKL threads por proceso (recomendado 1..num_cores)')
    args = p.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    # recomendación rápida de ajuste:
    print("Iniciando cálculo STD (CPU, matrix-free) —", datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    print("Recomendación: ajustar --tile_m y --batch_size para que m*tile_m*8 y batch_size*m*8 entren en RAM.", flush=True)
    calc_std_cpu_matrix_free(
        args.path_var_cor,
        args.path_pixeles,
        args.path_B,
        args.output_raster,
        args.pixel_size,
        batch_size=args.batch_size,
        tile_m=args.tile_m,
        num_workers=args.num_workers,
        blas_threads=args.blas_threads,
        save_pixels=True
    )
