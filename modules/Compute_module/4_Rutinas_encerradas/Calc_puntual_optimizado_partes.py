import sys
import numpy as np
import pandas as pd
import cupy as cp
from cupy import RawKernel
from datetime import datetime
import time
from tqdm import tqdm
import os
from numpy.lib.format import open_memmap

# Parámetros geodésicos
a = 6378137.0
b = 6356752.314140356

# Kernel CUDA optimizado: Clenshaw con constantes precalculadas
template = r'''
extern "C" __global__
void clenshaw_design(
    const double* __restrict__ R_norm,
    const double* __restrict__ c1,
    const double* __restrict__ c2,
    int N,
    const double* __restrict__ X,
    int n_obs,
    int n_malla,
    double* __restrict__ D
) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int j = blockDim.y * blockIdx.y + threadIdx.y;
    if (i >= n_obs || j >= n_malla) return;

    int idx = i * n_malla + j;
    double x = X[idx];

    double b_kp1 = 0.0, b_kp2 = 0.0, b_k = 0.0;
    for (int k = N - 1; k >= 0; --k) {
        b_k = c1[k] * x * b_kp1 - c2[k] * b_kp2 + R_norm[i * N + k];
        b_kp2 = b_kp1;
        b_kp1 = b_k;
    }
    D[idx] = b_k / (4.0 * 3.141592653589793);
}
'''
module = RawKernel(template, 'clenshaw_design')

def calcular_radio_gpu(lat_rad, h_gpu):
    sin_lat = cp.sin(lat_rad)
    cos_lat = cp.cos(lat_rad)
    num = a * b
    den = cp.sqrt((a * sin_lat)**2 + (b * cos_lat)**2)
    return num / den + h_gpu

def lat_elip_a_esf(lat_elip, a=6378137.0, b=6356752.314140356):
    f = (a - b) / a
    return cp.arctan((1 - f)**2 * cp.tan(lat_elip))

def calcular_cos_theta_gpu(obs_lat_elip, obs_lon, comp_lat_elip, comp_lon):
    obs_lat = lat_elip_a_esf(obs_lat_elip)
    comp_lat = lat_elip_a_esf(comp_lat_elip)
    dlon = comp_lon[None, :] - obs_lon[:, None]
    ct = (cp.sin(obs_lat[:, None]) * cp.sin(comp_lat[None, :]) +
          cp.cos(obs_lat[:, None]) * cp.cos(comp_lat[None, :]) * cp.cos(dlon))
    return cp.clip(ct, -1.0, 1.0)

def Filtro(filtro, n):
    if filtro == 'Cup':
        i = cp.arange(n, dtype=cp.float64)
        return ((1 - i/n)**2)*(1 + 2*i/n)
    else:
        return cp.ones(n, dtype=cp.float64)

def calcular_matriz_diseno_gpu(
    ruta_puntos, col_lat, col_lon, col_alt,
    ruta_malla, col_lat_malla, col_lon_malla,
    num_filas, para_pixeles, ruta_salida, filtro='None',
    tam_bloque=2000
):
    t0 = time.time()
    print(f"[INFO] Inicio: {datetime.now().strftime('%H:%M:%S')}", flush=True)
    print(f"[INFO] Leyendo informaciones: {datetime.now().strftime('%H:%M:%S')}", flush=True)

    # Leer observaciones
    pts = pd.read_csv(ruta_puntos, sep='\t')
    lat_o = np.radians(pts[col_lat].values)
    lon_o = np.radians(pts[col_lon].values)
    alt_o = np.zeros_like(lat_o) if col_alt == '0' else pts[col_alt].values

    # Leer malla
    m = pd.read_csv(ruta_malla, sep='\t')
    lat_m = np.radians(m[col_lat_malla].values)
    lon_m = np.radians(m[col_lon_malla].values)

    # Constantes precalculadas
    n = cp.arange(num_filas + 1, dtype=cp.float64)
    base = (2*n + 1)*(n+1) if para_pixeles == 0 else (2*n + 1)
    coefs_gpu = base * Filtro(filtro, num_filas + 1)

    # Precálculo c1 y c2
    k = np.arange(num_filas + 1)
    c1_gpu = cp.asarray(((2.0*k + 1.0) / (k + 1.0)).astype(np.float64))
    c2_gpu = cp.asarray(((k + 1.0) / (k + 2.0)).astype(np.float64))

    # Malla a GPU
    lat_m_gpu = cp.asarray(lat_m)
    lon_m_gpu = cp.asarray(lon_m)

    log_a = cp.log(cp.asarray(a, dtype=cp.float64))

    n_obs = len(lat_o)
    n_malla = len(lat_m)

    # Asegurar extensión .npy
    if not ruta_salida.lower().endswith('.npy'):
        ruta_salida += '.npy'

    # Crear/abrir .npy con cabecera usando open_memmap
    D_memmap = open_memmap(ruta_salida, mode='w+', dtype='float64', shape=(n_obs, n_malla))

    print(f"[INFO] Procesando en bloques de {tam_bloque} observaciones...", flush=True)
    for inicio in tqdm(range(0, n_obs, tam_bloque), desc="Procesando bloques"):
        fin = min(inicio + tam_bloque, n_obs)

        lat_b = cp.asarray(lat_o[inicio:fin])
        lon_b = cp.asarray(lon_o[inicio:fin])
        alt_b = cp.asarray(alt_o[inicio:fin])

        R1_b = calcular_radio_gpu(lat_b, alt_b)

        if para_pixeles == 0:
            R_norm_b = cp.exp((n[None, :] + 1)*log_a - (n[None, :] + 2)*cp.log(R1_b[:, None])) * coefs_gpu[None, :]
        else:
            R_norm_b = cp.exp((n[None, :] + 1)*log_a - (n[None, :] + 1)*cp.log(R1_b[:, None])) * coefs_gpu[None, :]

        X_b = calcular_cos_theta_gpu(lat_b, lon_b, lat_m_gpu, lon_m_gpu)

        n_b = X_b.shape[0]
        D_b = cp.zeros((n_b, n_malla), dtype=cp.float64)

        #Kernel config (prueba también (64,8) si te rinde mejor)
        block = (32, 32)
        grid  = ((n_b + block[0] - 1)//block[0], (n_malla + block[1] - 1)//block[1])

        module(grid, block,
            (R_norm_b, c1_gpu, c2_gpu, np.int32(num_filas + 1), X_b,
             np.int32(n_b), np.int32(n_malla), D_b))

        # Escribir bloque en el archivo .npy sin cargar todo en RAM
        D_memmap[inicio:fin, :] = cp.asnumpy(D_b)

        # Liberar GPU del bloque
        del lat_b, lon_b, alt_b, R1_b, R_norm_b, X_b, D_b
        cp.get_default_memory_pool().free_all_blocks()

    # Asegurar que todo quedó en disco
    D_memmap.flush()
    del D_memmap

    print(f"[INFO] Guardado completo en: {ruta_salida}", flush=True)
    print(f"[INFO] Tiempo total: {time.time() - t0:.2f} s", flush=True)
    return ruta_salida


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Cálculo matriz diseño GPU')
    parser.add_argument('ruta_puntos')
    parser.add_argument('col_lat')
    parser.add_argument('col_lon')
    parser.add_argument('col_alt')
    parser.add_argument('ruta_malla')
    parser.add_argument('col_lat_malla')
    parser.add_argument('col_lon_malla')
    parser.add_argument('num_filas', type=int)
    parser.add_argument('para_pixeles', type=int)
    parser.add_argument('ruta_salida')
    parser.add_argument('filtro')
    parser.add_argument('--tam_bloque', type=int, default=2000, help='Tamaño del bloque de observaciones para GPU')
    args = parser.parse_args()

    calcular_matriz_diseno_gpu(
        args.ruta_puntos,
        args.col_lat,
        args.col_lon,
        args.col_alt,
        args.ruta_malla,
        args.col_lat_malla,
        args.col_lon_malla,
        args.num_filas,
        args.para_pixeles,
        args.ruta_salida,
        args.filtro,
        args.tam_bloque
    )
