import argparse
import os
import sys
import time
import gc

import numpy as np
import pandas as pd
from datetime import datetime
from rasterio.transform import from_origin
from rasterio.crs import CRS
import rasterio

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

UO = 62636860.850
WO = 62636853.4
gamma_a = 978032.67715
gamma_b = 983218.63685
c1 = 0.0053024
c2 = 0.0000058
f = 0.00335281068118
m_par = 0.00344978600308
a = 6378137.0
b = 6356752.3141
e2 = 0.00669438002290


def gravedad_teorica_somigliana(lat):
    lat_r = np.radians(lat)
    num = (a * gamma_a * np.cos(lat_r) ** 2 + b * gamma_b * np.sin(lat_r) ** 2)
    den = np.sqrt(a ** 2 * np.cos(lat_r) ** 2 + b ** 2 * np.sin(lat_r) ** 2)
    return num / den


def gamma_h(lat, gamma_0, h):
    lat_r = np.radians(lat)
    return gamma_0 * (1 - (2 / a) * (1 + f + m_par - 2 * f * np.sin(lat_r) ** 2) * h
                      + (3 / (a ** 2)) * (h ** 2))


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
    lat_vals = df['Latitud'].values
    lon_vals = df['Longitud'].values
    std_vals = df['Std_Zeta'].values
    for k in range(len(df)):
        i = lat_idx[lat_vals[k]]
        j = lon_idx[lon_vals[k]]
        raster[i, j] = std_vals[k]
    transform = from_origin(longitudes[0] - pixel_size / 2,
                            latitudes[0] + pixel_size / 2,
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
    print(f"Raster generado: {output_filename} {datetime.now().strftime('%H:%M:%S')}",
          flush=True)


def _import_generador():
    try:
        from Generador_de_RasterInsitu import preparar_generador, iter_bloques_diseno
        return preparar_generador, iter_bloques_diseno
    except Exception as e:
        raise RuntimeError(
            f"No se pudo importar el generador in-situ desde "
            f"Generador_de_Raster1.py: {e}"
        )


def _get_free_vram():
    try:
        import cupy as cp
        free_b, _ = cp.cuda.Device(0).mem_info
        return int(free_b)
    except Exception:
        return 0


def _std_in_situ_gpu(prep, path_M, tam_bloque, tile_m):
    import cupy as cp
    n_obs = prep["n_obs"]
    n_malla = prep["n_malla"]

    M_mm = np.load(path_M, mmap_mode='r')
    if M_mm.shape != (n_malla, n_malla):
        raise ValueError(
            f"M tiene shape {M_mm.shape}, se esperaba ({n_malla}, {n_malla})"
        )

    free_vram = _get_free_vram()
    bytes_M = int(M_mm.nbytes)
    usar_M_completa = (bytes_M * 3) < free_vram

    if usar_M_completa:
        print(f"[in-situ] M completa en VRAM: {bytes_M/1e9:.2f} GB "
              f"(libre {free_vram/1e9:.2f} GB)", flush=True)
        M_gpu = cp.asarray(M_mm, dtype=cp.float64)
        M_mm = None
        tile_efectivo = n_malla
    else:
        print(f"[in-situ] M por tiles: {bytes_M/1e9:.2f} GB. "
              f"tile_m={tile_m}", flush=True)
        M_gpu = None
        tile_efectivo = int(tile_m)

    resultado = np.empty(n_obs, dtype=np.float64)
    tam_bloque = int(tam_bloque)
    n_bloques = (n_obs + tam_bloque - 1) // tam_bloque
    total_elems = n_obs * n_malla

    preparar_generador, iter_bloques_diseno = _import_generador()

    t0 = time.perf_counter()
    t_last = t0
    completados = 0
    elems_done = 0

    for i0, i1, B_blk in iter_bloques_diseno(prep, tam_bloque=tam_bloque):
        n_b = i1 - i0

        if usar_M_completa:
            tmp = cp.matmul(B_blk, M_gpu)
            diag = cp.einsum('ij,ij->i', tmp, B_blk)
            del tmp
        else:
            acc = cp.zeros(n_b, dtype=cp.float64)
            for j0 in range(0, n_malla, tile_efectivo):
                j1 = min(j0 + tile_efectivo, n_malla)
                M_tile = cp.asarray(M_mm[:, j0:j1], dtype=cp.float64)
                tmp = cp.matmul(B_blk, M_tile)
                b_tile = B_blk[:, j0:j1]
                acc += cp.einsum('ij,ij->i', tmp, b_tile)
                del M_tile, tmp, b_tile
                cp.get_default_memory_pool().free_all_blocks()
            diag = acc

        diag = cp.maximum(diag, 0.0)
        stds = cp.sqrt(diag)
        resultado[i0:i1] = cp.asnumpy(stds)

        del B_blk, diag, stds
        cp.get_default_memory_pool().free_all_blocks()

        completados += 1
        elems_done += n_b * n_malla

        now = time.perf_counter()
        if now - t_last >= 5.0 or completados == n_bloques:
            elapsed = now - t0
            pct = elems_done * 100.0 / total_elems if total_elems else 100.0
            rate = elems_done / elapsed if elapsed > 0 else 0.0
            eta = (total_elems - elems_done) / rate if rate > 0 else 0.0
            print(f"[{completados:>{len(str(n_bloques))}}/{n_bloques}] "
                  f"{pct:6.2f}% ETA {eta/60:8.2f} min", flush=True)
            t_last = now

    if M_gpu is not None:
        del M_gpu
        cp.get_default_memory_pool().free_all_blocks()

    return resultado


def _std_legacy(path_M, path_B, batch_size, tile_m):
    print("[legacy] Abriendo memmaps...", flush=True)
    M_mm = np.load(path_M, mmap_mode='r')
    B_mm = np.load(path_B, mmap_mode='r')
    if M_mm.shape[0] != M_mm.shape[1]:
        raise ValueError("M debe ser cuadrada")
    if M_mm.shape[0] != B_mm.shape[1]:
        raise ValueError(f"Mismatch shapes M={M_mm.shape} B={B_mm.shape}")
    m = M_mm.shape[0]
    n = B_mm.shape[0]

    std_T = np.empty(n, dtype=np.float64)
    total_elems = n * m
    elems_done = 0
    t0 = time.perf_counter()
    t_last = t0
    n_batches = (n + batch_size - 1) // batch_size
    batch_idx = 0

    for i0 in range(0, n, batch_size):
        i1 = min(i0 + batch_size, n)
        B_batch = np.asarray(B_mm[i0:i1, :], dtype=np.float64, order='C')
        acc = np.zeros(i1 - i0, dtype=np.float64)

        for j0 in range(0, m, tile_m):
            j1 = min(j0 + tile_m, m)
            M_tile = np.asarray(M_mm[:, j0:j1], dtype=np.float64, order='F')
            tmp = B_batch.dot(M_tile)
            b_tile = B_batch[:, j0:j1]
            acc += np.einsum('ij,ij->i', tmp, b_tile)
            del M_tile, tmp, b_tile
            gc.collect()

        acc = np.where(acc < 0, 0.0, acc)
        std_T[i0:i1] = np.sqrt(acc)
        del B_batch
        gc.collect()

        batch_idx += 1
        elems_done += (i1 - i0) * m
        now = time.perf_counter()
        if now - t_last >= 5.0 or batch_idx == n_batches:
            elapsed = now - t0
            pct = elems_done * 100.0 / total_elems if total_elems else 100.0
            rate = elems_done / elapsed if elapsed > 0 else 0.0
            eta = (total_elems - elems_done) / rate if rate > 0 else 0.0
            print(f"[{batch_idx}/{n_batches}] {pct:6.2f}% "
                  f"ETA {eta/60:8.2f} min", flush=True)
            t_last = now

    return std_T


def calc_std(path_var_cor_npy,
             path_pixeles,
             output_raster,
             pixel_size,
             legacy=False,
             path_B=None,
             batch_size=500,
             tile_m=512,
             save_pixels=True,
             ruta_puntos=None,
             col_lat=None, col_lon=None, col_alt=None,
             ruta_malla=None,
             col_lat_malla=None, col_lon_malla=None,
             num_filas=None, para_pixeles=1,
             filtro="Cup",
             tam_bloque=2000):

    t_start = datetime.now()
    print("=" * 78)
    print(f"Iniciando calculo STD - {t_start.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 78, flush=True)

    if legacy:
        print("Modo                : LEGACY (.npy)")
        if path_B is None:
            raise ValueError("Modo legacy requiere --path_B")
        if not os.path.isfile(path_B):
            raise FileNotFoundError(path_B)
        if not os.path.isfile(path_var_cor_npy):
            raise FileNotFoundError(path_var_cor_npy)

        B_mm = np.load(path_B, mmap_mode='r')
        n = B_mm.shape[0]
        del B_mm
        print(f"n_obs               : {n:,}")
        print(f"batch_size          : {batch_size:,}")
        print(f"tile_m              : {tile_m:,}")

        std_T = _std_legacy(path_var_cor_npy, path_B, batch_size, tile_m)
    else:
        print("Modo                : IN-SITU con GPU")
        for req, name in [(ruta_puntos, "--ruta-puntos"),
                          (ruta_malla, "--ruta-malla"),
                          (num_filas, "--num-filas"),
                          (col_lat, "--col-lat"),
                          (col_lon, "--col-lon"),
                          (col_alt, "--col-alt"),
                          (col_lat_malla, "--col-lat-malla"),
                          (col_lon_malla, "--col-lon-malla")]:
            if req is None:
                raise ValueError(f"Falta argumento obligatorio: {name}")
        if not os.path.isfile(ruta_puntos):
            raise FileNotFoundError(ruta_puntos)
        if not os.path.isfile(ruta_malla):
            raise FileNotFoundError(ruta_malla)
        if not os.path.isfile(path_var_cor_npy):
            raise FileNotFoundError(path_var_cor_npy)

        preparar_generador, _ = _import_generador()

        t_prep = time.perf_counter()
        prep = preparar_generador(
            ruta_puntos, col_lat, col_lon, col_alt,
            ruta_malla, col_lat_malla, col_lon_malla,
            num_filas, para_pixeles, filtro,
        )
        print(f"n_obs               : {prep['n_obs']:,}")
        print(f"n_malla             : {prep['n_malla']:,}")
        print(f"num_filas           : {num_filas}")
        print(f"para_pixeles        : {para_pixeles}")
        print(f"filtro              : {filtro}")
        print(f"tam_bloque          : {tam_bloque:,}")
        print(f"Preparacion         : {time.perf_counter() - t_prep:.2f}s")

        std_T = _std_in_situ_gpu(prep, path_var_cor_npy, tam_bloque, tile_m)
        n = len(std_T)

    print("[post] Procesando DataFrame y raster...", flush=True)
    df = pd.read_csv(path_pixeles, sep='\t', header=0)
    if len(df) != n:
        print(f"[WARN] puntos={len(df)} != filas calculadas={n}. "
              f"Se ajusta por minimo.", flush=True)
        minlen = min(len(df), n)
        df = df.iloc[:minlen].copy()
        std_T = std_T[:minlen]

    df["Std_T"] = std_T.astype(np.float32)
    df = calculo_desv_modelo(df)
    generar_raster(df, output_raster, pixel_size / 60.0)
    if save_pixels:
        df.to_csv(path_pixeles, sep="\t", index=False)

    t_end = datetime.now()
    print("=" * 78)
    print(f"Proceso completado : {t_end.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Duracion           : {t_end - t_start}")
    print("=" * 78, flush=True)


def parse_args():
    p = argparse.ArgumentParser(
        description="Calculo STD del potencial (T) en float64."
    )
    p.add_argument('path_var_cor', help=".npy de la matriz M (m x m)")
    p.add_argument('path_pixeles', help="archivo .tsv con puntos")
    p.add_argument('output_raster', help="ruta salida GeoTIFF")
    p.add_argument('pixel_size', type=float,
                   help="tamano de pixel (minutos)")

    p.add_argument('--legacy', action='store_true', default=False,
                   help="Modo legacy leyendo B desde .npy. "
                        "Por defecto se usa modo in-situ.")

    p.add_argument('--path_B', type=str, default=None,
                   help=".npy de la matriz B (n x m) - solo legacy")
    p.add_argument('--batch_size', type=int, default=500)
    p.add_argument('--tile_m', type=int, default=512)

    p.add_argument('--ruta-puntos', '--ruta_puntos', type=str, default=None,
                   dest='ruta_puntos')
    p.add_argument('--col-lat', default=None, dest='col_lat')
    p.add_argument('--col-lon', default=None, dest='col_lon')
    p.add_argument('--col-alt', default=None, dest='col_alt')
    p.add_argument('--ruta-malla', '--ruta_malla', type=str, default=None,
                   dest='ruta_malla')
    p.add_argument('--col-lat-malla', default=None, dest='col_lat_malla')
    p.add_argument('--col-lon-malla', default=None, dest='col_lon_malla')
    p.add_argument('--num-filas', '--num_filas', type=int, default=None,
                   dest='num_filas')
    p.add_argument('--para-pixeles', '--para_pixeles', type=int, default=1,
                   dest='para_pixeles')
    p.add_argument('--filtro', default='Cup')
    p.add_argument('--tam-bloque', '--tam_bloque', type=int, default=2000,
                   dest='tam_bloque')

    p.add_argument('--no-save-pixels', action='store_true', default=False)

    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()
    calc_std(
        path_var_cor_npy=args.path_var_cor,
        path_pixeles=args.path_pixeles,
        output_raster=args.output_raster,
        pixel_size=args.pixel_size,
        legacy=args.legacy,
        path_B=args.path_B,
        batch_size=args.batch_size,
        tile_m=args.tile_m,
        save_pixels=(not args.no_save_pixels),
        ruta_puntos=args.ruta_puntos,
        col_lat=args.col_lat,
        col_lon=args.col_lon,
        col_alt=args.col_alt,
        ruta_malla=args.ruta_malla,
        col_lat_malla=args.col_lat_malla,
        col_lon_malla=args.col_lon_malla,
        num_filas=args.num_filas,
        para_pixeles=args.para_pixeles,
        filtro=args.filtro,
        tam_bloque=args.tam_bloque,
    )