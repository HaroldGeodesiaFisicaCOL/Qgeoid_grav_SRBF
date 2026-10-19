import os
import sys
import time
import argparse
import threading
import queue
import platform
import subprocess
import gc

CPU_COUNT = os.cpu_count() or 1
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, str(CPU_COUNT))

import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import from_origin
from rasterio.crs import CRS


def get_free_ram_mb():
    try:
        import psutil
        return psutil.virtual_memory().available / 1024**2
    except Exception:
        pass
    if platform.system() == "Windows":
        try:
            import ctypes
            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]
            s = MEMORYSTATUSEX()
            s.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(s))
            return s.ullAvailPhys / 1024**2
        except Exception:
            pass
    try:
        with open("/proc/meminfo", "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) / 1024
    except Exception:
        pass
    return None


def detect_gpu():
    try:
        import cupy as cp
        n = cp.cuda.runtime.getDeviceCount()
        if n < 1:
            return None
        dev = cp.cuda.Device(0)
        free_b, total_b = dev.mem_info
        props = cp.cuda.runtime.getDeviceProperties(0)
        name = props["name"]
        if isinstance(name, bytes):
            name = name.decode(errors="replace")
        t = cp.empty(1, dtype=cp.float64)
        t.fill(0.0)
        dev.synchronize()
        del t
        return {
            "nombre": name,
            "vram_libre_b": int(free_b),
            "vram_total_b": int(total_b),
            "vram_libre_gb": free_b / 1e9,
            "vram_total_gb": total_b / 1e9,
        }
    except Exception:
        return None


def detect_disk_media_type(path):
    if platform.system() != "Windows":
        return None
    try:
        drive = os.path.splitdrive(os.path.abspath(path))[0].rstrip(":")
        ps = (
            f"$dl='{drive}'; "
            f"$p=Get-Partition -DriveLetter $dl -ErrorAction SilentlyContinue; "
            f"if($p){{ $d=Get-PhysicalDisk -DeviceNumber $p.DiskNumber -ErrorAction SilentlyContinue; "
            f"if($d){{Write-Output $d.MediaType; Write-Output $d.BusType}}}}"
        )
        out = subprocess.check_output(
            ["powershell", "-NoProfile", "-Command", ps],
            stderr=subprocess.DEVNULL, timeout=10,
        ).decode(errors="replace").strip().splitlines()
        if not out:
            return None
        media = out[0].strip().lower()
        bus = out[1].strip().lower() if len(out) > 1 else ""
        if "nvme" in bus or "nvme" in media:
            return "NVMe"
        if "ssd" in media:
            return "SSD"
        if "hdd" in media or "unspecified" in media or "fixed" in media:
            return "HDD"
        return media.upper() if media else None
    except Exception:
        return None


_KERNEL_SRC = r'''
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

_cuda_kernel = None


def _get_kernel():
    global _cuda_kernel
    if _cuda_kernel is None:
        from cupy import RawKernel
        _cuda_kernel = RawKernel(_KERNEL_SRC, "clenshaw_design")
    return _cuda_kernel


_A = 6378137.0
_B = 6356752.314140356


def _filtro_cupy(filtro, n):
    import cupy as cp
    if filtro == "Cup":
        i = cp.arange(n, dtype=cp.float64)
        return ((1 - i / n) ** 2) * (1 + 2 * i / n)
    return cp.ones(n, dtype=cp.float64)


def _radio_gpu(lat_rad, h_gpu):
    import cupy as cp
    sin_lat = cp.sin(lat_rad)
    cos_lat = cp.cos(lat_rad)
    num = _A * _B
    den = cp.sqrt((_A * sin_lat) ** 2 + (_B * cos_lat) ** 2)
    return num / den + h_gpu


def _lat_elip_a_esf(lat_elip):
    import cupy as cp
    f = (_A - _B) / _A
    return cp.arctan((1 - f) ** 2 * cp.tan(lat_elip))


def _cos_theta_gpu(obs_lat_elip, obs_lon, comp_lat_elip, comp_lon):
    import cupy as cp
    obs_lat = _lat_elip_a_esf(obs_lat_elip)
    comp_lat = _lat_elip_a_esf(comp_lat_elip)
    dlon = comp_lon[None, :] - obs_lon[:, None]
    ct = (cp.sin(obs_lat[:, None]) * cp.sin(comp_lat[None, :]) +
          cp.cos(obs_lat[:, None]) * cp.cos(comp_lat[None, :]) * cp.cos(dlon))
    return cp.clip(ct, -1.0, 1.0)


def preparar_generador(ruta_puntos, col_lat, col_lon, col_alt,
                       ruta_malla, col_lat_malla, col_lon_malla,
                       num_filas, para_pixeles, filtro):
    import cupy as cp

    pts = pd.read_csv(ruta_puntos, sep="\t")
    lat_o = np.radians(pts[col_lat].values)
    lon_o = np.radians(pts[col_lon].values)
    alt_o = (np.zeros_like(lat_o) if col_alt == "0"
             else pts[col_alt].values.astype(np.float64))

    m = pd.read_csv(ruta_malla, sep="\t")
    lat_m = np.radians(m[col_lat_malla].values)
    lon_m = np.radians(m[col_lon_malla].values)

    n = cp.arange(num_filas + 1, dtype=cp.float64)
    base = (2 * n + 1) * (n + 1) if para_pixeles == 0 else (2 * n + 1)
    coefs_gpu = base * _filtro_cupy(filtro, num_filas + 1)

    k = np.arange(num_filas + 1)
    c1_gpu = cp.asarray(((2.0 * k + 1.0) / (k + 1.0)).astype(np.float64))
    c2_gpu = cp.asarray(((k + 1.0) / (k + 2.0)).astype(np.float64))

    lat_m_gpu = cp.asarray(lat_m)
    lon_m_gpu = cp.asarray(lon_m)
    log_a = cp.log(cp.asarray(_A, dtype=cp.float64))

    return {
        "lat_o": lat_o,
        "lon_o": lon_o,
        "alt_o": alt_o,
        "n_obs": int(len(lat_o)),
        "n_malla": int(len(lat_m)),
        "lat_m_gpu": lat_m_gpu,
        "lon_m_gpu": lon_m_gpu,
        "n": n,
        "coefs_gpu": coefs_gpu,
        "c1_gpu": c1_gpu,
        "c2_gpu": c2_gpu,
        "log_a": log_a,
        "num_filas": int(num_filas),
        "para_pixeles": int(para_pixeles),
    }


def iter_bloques_diseno(prep, tam_bloque=2000):
    import cupy as cp
    lat_o = prep["lat_o"]
    lon_o = prep["lon_o"]
    alt_o = prep["alt_o"]
    n_obs = prep["n_obs"]
    n_malla = prep["n_malla"]
    lat_m_gpu = prep["lat_m_gpu"]
    lon_m_gpu = prep["lon_m_gpu"]
    n = prep["n"]
    coefs_gpu = prep["coefs_gpu"]
    c1_gpu = prep["c1_gpu"]
    c2_gpu = prep["c2_gpu"]
    log_a = prep["log_a"]
    N = prep["num_filas"] + 1
    para_pixeles = prep["para_pixeles"]

    kernel = _get_kernel()

    for i0 in range(0, n_obs, tam_bloque):
        i1 = min(i0 + tam_bloque, n_obs)

        lat_b = cp.asarray(lat_o[i0:i1])
        lon_b = cp.asarray(lon_o[i0:i1])
        alt_b = cp.asarray(alt_o[i0:i1])

        R1_b = _radio_gpu(lat_b, alt_b)

        if para_pixeles == 0:
            R_norm_b = cp.exp((n[None, :] + 1) * log_a
                              - (n[None, :] + 2) * cp.log(R1_b[:, None])) \
                       * coefs_gpu[None, :]
        else:
            R_norm_b = cp.exp((n[None, :] + 1) * log_a
                              - (n[None, :] + 1) * cp.log(R1_b[:, None])) \
                       * coefs_gpu[None, :]

        X_b = _cos_theta_gpu(lat_b, lon_b, lat_m_gpu, lon_m_gpu)
        n_b = X_b.shape[0]
        D_b = cp.zeros((n_b, n_malla), dtype=cp.float64)

        block = (32, 32)
        grid = ((n_b + block[0] - 1) // block[0],
                (n_malla + block[1] - 1) // block[1])

        kernel(grid, block,
               (R_norm_b, c1_gpu, c2_gpu, np.int32(N), X_b,
                np.int32(n_b), np.int32(n_malla), D_b))

        yield i0, i1, D_b

        del lat_b, lon_b, alt_b, R1_b, R_norm_b, X_b, D_b
        cp.get_default_memory_pool().free_all_blocks()


def _generar_in_situ_gpu(prep, parametros, tam_bloque, resultado):
    import cupy as cp
    n_obs = prep["n_obs"]
    n_malla = prep["n_malla"]
    total_bytes = n_obs * n_malla * 8
    tam_bloque = int(tam_bloque)
    n_bloques = (n_obs + tam_bloque - 1) // tam_bloque

    params_gpu = cp.asarray(parametros, dtype=cp.float64)

    t0 = time.perf_counter()
    t_last = t0
    bytes_done = 0
    completados = 0

    for i0, i1, A_blk in iter_bloques_diseno(prep, tam_bloque=tam_bloque):
        n_b = i1 - i0
        res_blk = cp.dot(A_blk, params_gpu)
        resultado[i0:i1] = cp.asnumpy(res_blk)
        del res_blk, A_blk

        completados += 1
        bytes_done += n_b * n_malla * 8

        now = time.perf_counter()
        if now - t_last >= 5.0 or completados == n_bloques:
            elapsed = now - t0
            mbps = bytes_done / 1e6 / elapsed if elapsed > 0 else 0.0
            pct = bytes_done * 100.0 / total_bytes if total_bytes else 100.0
            rate = bytes_done / elapsed if elapsed > 0 else 0.0
            eta = (total_bytes - bytes_done) / rate if rate > 0 else 0.0
            print(f"[{completados:>{len(str(n_bloques))}}/{n_bloques}] "
                  f"{pct:6.2f}% {mbps:8.1f} MB/s-equiv ETA {eta/60:8.2f} min",
                  flush=True)
            t_last = now

    cp.get_default_memory_pool().free_all_blocks()


def leer_header_npy(path):
    with open(path, "rb") as f:
        version = np.lib.format.read_magic(f)
        if version == (1, 0):
            shape, fortran, dtype = np.lib.format.read_array_header_1_0(f)
        elif version in ((2, 0), (3, 0)):
            shape, fortran, dtype = np.lib.format.read_array_header_2_0(f)
        else:
            raise ValueError(f"Version .npy no soportada: {version}")
        offset = f.tell()
    return tuple(shape), np.dtype(dtype), offset, bool(fortran)


def _open_sequential(path):
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0)
    if platform.system() == "Windows":
        flags |= getattr(os, "O_SEQUENTIAL", 0)
    return os.fdopen(os.open(path, flags), "rb", buffering=0)


def _reader_worker(path, offset, row_bytes, n_cols, dtype, n_rows, chunk_rows,
                   work_q, free_q, ready_q, buffers, stop_event):
    f = None
    try:
        f = _open_sequential(path)
        while not stop_event.is_set():
            try:
                block_index, i0 = work_q.get_nowait()
            except queue.Empty:
                break
            buffer_index = None
            try:
                while not stop_event.is_set():
                    try:
                        buffer_index = free_q.get(timeout=0.25)
                        break
                    except queue.Empty:
                        continue
                if stop_event.is_set():
                    break
                n = min(chunk_rows, n_rows - i0)
                elems = n * n_cols
                target = buffers[buffer_index][:elems]
                f.seek(offset + i0 * row_bytes, os.SEEK_SET)
                need = n * row_bytes
                view = memoryview(target).cast("B")[:need]
                read_total = 0
                while read_total < need:
                    got = f.readinto(view[read_total:])
                    if not got:
                        raise IOError(f"Lectura incompleta bloque {block_index}: "
                                      f"{read_total}/{need}")
                    read_total += got
                ready_q.put(("data", block_index, i0, n, buffer_index))
                buffer_index = None
            except Exception:
                if buffer_index is not None:
                    free_q.put(buffer_index)
                raise
    except Exception as e:
        stop_event.set()
        try:
            ready_q.put(("error", e, None, None, None), timeout=1.0)
        except Exception:
            pass
    finally:
        if f is not None:
            try:
                f.close()
            except Exception:
                pass
        try:
            ready_q.put(("done", None, None, None, None), timeout=1.0)
        except Exception:
            pass


def _make_pinned_array(cp, n_elements, dtype):
    nbytes = int(n_elements) * np.dtype(dtype).itemsize
    mem = cp.cuda.alloc_pinned_memory(nbytes)
    return mem, np.frombuffer(mem, dtype=dtype, count=n_elements)


def _generar_legacy(path_npy, offset, dtype, n_filas, n_cols, parametros,
                    filas_por_bloque, n_bloques, n_lectores, host_buffers,
                    resultado):
    import cupy as cp
    total_elems = filas_por_bloque * n_cols
    host_memories = []
    host_arrays = []
    pinned_ok = True
    try:
        for _ in range(host_buffers):
            mem, arr = _make_pinned_array(cp, total_elems, dtype)
            host_memories.append(mem)
            host_arrays.append(arr)
    except Exception as e:
        pinned_ok = False
        print(f"[AVISO] Memoria pinned no disponible: {e}")
        host_memories = []
        host_arrays = [np.empty(total_elems, dtype=dtype)
                       for _ in range(host_buffers)]

    streams = [cp.cuda.Stream(non_blocking=True) for _ in range(host_buffers)]
    device_arrays = [cp.empty((filas_por_bloque, n_cols), dtype=cp.float64)
                     for _ in range(host_buffers)]
    device_results = [cp.empty(filas_por_bloque, dtype=cp.float64)
                      for _ in range(host_buffers)]
    params_gpu = cp.asarray(parametros, dtype=cp.float64)

    work_q = queue.Queue()
    for i0 in range(0, n_filas, filas_por_bloque):
        work_q.put((i0 // filas_por_bloque, i0))
    free_q = queue.Queue(maxsize=host_buffers)
    for bi in range(host_buffers):
        free_q.put(bi)
    ready_q = queue.Queue(maxsize=max(host_buffers + n_lectores + 2, 8))
    stop_event = threading.Event()

    threads = []
    for _ in range(n_lectores):
        t = threading.Thread(
            target=_reader_worker,
            args=(path_npy, offset, int(dtype.itemsize) * n_cols, n_cols,
                  dtype, n_filas, filas_por_bloque, work_q, free_q, ready_q,
                  host_arrays, stop_event),
            daemon=True,
        )
        t.start()
        threads.append(t)

    pending = {}
    received = 0
    completed = 0
    bytes_done = 0
    total_bytes = n_filas * n_cols * 8
    t0 = time.perf_counter()
    t_last = t0

    try:
        while received < n_bloques:
            item = ready_q.get()
            kind = item[0]
            if kind == "error":
                stop_event.set()
                raise RuntimeError(f"Error de lectura: {item[1]}")
            if kind == "done":
                continue
            _, block_index, i0, n, buffer_index = item
            received += 1
            stream = streams[buffer_index]
            stream.synchronize()
            host_arr = host_arrays[buffer_index][:n * n_cols].reshape(n, n_cols)
            dev_arr = device_arrays[buffer_index]
            dev_res = device_results[buffer_index]
            with stream:
                dev_arr[:n].set(host_arr, stream=stream)
                cp.matmul(dev_arr[:n], params_gpu, out=dev_res[:n])
            pending[buffer_index] = (stream, dev_res, i0, n, buffer_index)
            bytes_done += n * n_cols * 8

            if len(pending) >= host_buffers:
                oldest = min(pending, key=lambda k: pending[k][2])
                st, dr, oi, on, bi = pending.pop(oldest)
                st.synchronize()
                resultado[oi:oi + on] = cp.asnumpy(dr[:on])
                free_q.put(bi)
                completed += 1

            now = time.perf_counter()
            if now - t_last >= 5.0 or received == n_bloques:
                elapsed = now - t0
                mbps = bytes_done / 1e6 / elapsed if elapsed > 0 else 0.0
                pct = bytes_done * 100.0 / total_bytes if total_bytes else 100.0
                rate = bytes_done / elapsed if elapsed > 0 else 0.0
                eta = (total_bytes - bytes_done) / rate if rate > 0 else 0.0
                print(f"[{completed:>{len(str(n_bloques))}}/{n_bloques}] "
                      f"{pct:6.2f}% {mbps:8.1f} MB/s ETA {eta/60:8.2f} min",
                      flush=True)
                t_last = now

        while pending:
            oldest = min(pending, key=lambda k: pending[k][2])
            st, dr, oi, on, bi = pending.pop(oldest)
            st.synchronize()
            resultado[oi:oi + on] = cp.asnumpy(dr[:on])
            free_q.put(bi)
            completed += 1
        stop_event.set()
        for t in threads:
            t.join(timeout=30)
        cp.cuda.Device().synchronize()
    except Exception:
        stop_event.set()
        for t in threads:
            t.join(timeout=5)
        raise
    finally:
        try:
            for s in streams:
                s.synchronize()
        except Exception:
            pass
        try:
            del params_gpu
        except Exception:
            pass
        try:
            del device_arrays
        except Exception:
            pass
        try:
            del device_results
        except Exception:
            pass
        try:
            del streams
        except Exception:
            pass
        try:
            cp.get_default_memory_pool().free_all_blocks()
            cp.get_default_pinned_memory_pool().free_all_blocks()
        except Exception:
            pass

    if not pinned_ok:
        print("[AVISO] Se proceso con buffers host no-pinned.")


def _auto_config_legacy(path_npy, n_filas, n_cols, bytes_override,
                        n_lectores_override, chunk_rows_override):
    row_bytes = n_cols * 8
    free_ram_mb = get_free_ram_mb()
    gpu = detect_gpu()
    media = detect_disk_media_type(path_npy)

    if chunk_rows_override is not None:
        filas = max(1, min(int(chunk_rows_override), n_filas))
    elif bytes_override is not None:
        filas = max(1, min(int(bytes_override) // row_bytes, n_filas))
    else:
        target_mb = 256.0
        if media == "SSD":
            target_mb = 512.0
        elif media == "NVMe":
            target_mb = 768.0
        if free_ram_mb is not None:
            target_mb = min(target_mb,
                            max(256.0, min(2048.0, free_ram_mb * 0.10)) / 3.0)
        if gpu is not None:
            target_mb = min(target_mb,
                            max(256.0, gpu["vram_libre_b"] / 1024**2 * 0.55) / 3.0)
        target_mb = max(32.0, min(1024.0, target_mb))
        filas = max(1, min(int((target_mb * 1024**2) // row_bytes), n_filas))

    if n_lectores_override is not None:
        n_lectores = max(1, int(n_lectores_override))
    else:
        n_lectores = 1 if media in (None, "HDD") else 2

    total_bloques = (n_filas + filas - 1) // filas
    n_lectores = min(n_lectores, total_bloques)
    host_buffers = 3 if gpu is not None else 1
    if gpu is not None and media == "HDD":
        n_lectores = 1

    return {
        "free_ram_mb": free_ram_mb,
        "gpu": gpu,
        "media_type": media,
        "n_lectores": n_lectores,
        "host_buffers": host_buffers,
        "filas_por_bloque": filas,
        "bytes_bloque": filas * row_bytes,
        "total_bloques": total_bloques,
    }


def _cargar_puntos(path):
    puntos = pd.read_csv(path, sep="\t")
    if "Latitud" not in puntos.columns or "Longitud" not in puntos.columns:
        raise ValueError("El archivo de puntos debe contener Latitud y Longitud.")
    return puntos


def _preparar_raster(puntos, resultado, param_nom, raster_puntos_centrales):
    nombre_col = {"T": "Potencial_Perturbador",
                  "Z": "Anomalia_altura",
                  "Std": "Std_Qgeoide"}.get(param_nom, param_nom)

    # NUEVO: conservar el resultado para las etapas posteriores
    puntos[nombre_col] = np.asarray(resultado, dtype=np.float64)

    puntos.to_csv(
        raster_puntos_centrales,
        sep="\t",
        index=False,
        float_format="%.20f"
    )

    print(f"Resultados actualizados en: {raster_puntos_centrales}")

    lat = puntos["Latitud"].to_numpy(dtype=np.float64, copy=False)
    lon = puntos["Longitud"].to_numpy(dtype=np.float64, copy=False)

    if len(resultado) != len(puntos):
        raise ValueError(f"Matriz tiene {len(resultado):,} filas, "
                         f"puntos {len(puntos):,}.")

    order = np.lexsort((lon, -lat))
    lat_s = lat[order]
    lon_s = lon[order]
    val_s = np.asarray(resultado, dtype=np.float64)[order]

    unique_lat, lat_counts = np.unique(lat_s, return_counts=True)
    unique_lon = np.unique(lon_s)

    regular = (len(unique_lat) * len(unique_lon) == len(puntos)
               and np.all(lat_counts == len(unique_lon)))

    if regular:
        grid = val_s.reshape(len(unique_lat), len(unique_lon))
        latitudes = unique_lat[::-1]
        longitudes = unique_lon
    else:
        puntos = puntos.copy()
        puntos = puntos.sort_values(
            ["Latitud", "Longitud"],
            ascending=[False, True]
        )

        pivot = puntos.pivot(
            index="Latitud",
            columns="Longitud",
            values=nombre_col
        )

        grid = pivot.values
        latitudes = pivot.index.to_numpy(dtype=np.float64)
        longitudes = pivot.columns.to_numpy(dtype=np.float64)
        grid = grid.astype(np.float64, copy=False)

    return grid, latitudes, longitudes
    nombre_col = {"T": "Potencial_Perturbador",
                  "Z": "Anomalia_altura",
                  "Std": "Std_Qgeoide"}.get(param_nom, param_nom)
    lat = puntos["Latitud"].to_numpy(dtype=np.float64, copy=False)
    lon = puntos["Longitud"].to_numpy(dtype=np.float64, copy=False)
    if len(resultado) != len(puntos):
        raise ValueError(f"Matriz tiene {len(resultado):,} filas, "
                         f"puntos {len(puntos):,}.")
    order = np.lexsort((lon, -lat))
    lat_s = lat[order]
    lon_s = lon[order]
    val_s = np.asarray(resultado, dtype=np.float64)[order]
    unique_lat, lat_counts = np.unique(lat_s, return_counts=True)
    unique_lon = np.unique(lon_s)
    regular = (len(unique_lat) * len(unique_lon) == len(puntos)
               and np.all(lat_counts == len(unique_lon)))
    if regular:
        grid = val_s.reshape(len(unique_lat), len(unique_lon))
        latitudes = unique_lat[::-1]
        longitudes = unique_lon
    else:
        puntos = puntos.copy()
        puntos[nombre_col] = resultado
        puntos = puntos.sort_values(["Latitud", "Longitud"],
                                    ascending=[False, True])
        pivot = puntos.pivot(index="Latitud", columns="Longitud",
                             values=nombre_col)
        grid = pivot.values
        latitudes = pivot.index.to_numpy(dtype=np.float64)
        longitudes = pivot.columns.to_numpy(dtype=np.float64)
        grid = grid.astype(np.float64, copy=False)
    return grid, latitudes, longitudes


def _write_raster(nombre_raster, grid, latitudes, longitudes):
    if len(latitudes) < 2 or len(longitudes) < 2:
        raise ValueError("Puntos insuficientes para definir la grilla.")
    lat_step = abs(float(latitudes[0]) - float(latitudes[1]))
    lon_step = abs(float(longitudes[1]) - float(longitudes[0]))
    if lat_step <= 0 or lon_step <= 0:
        raise ValueError("Resolucion de grilla invalida.")
    west = float(longitudes[0]) - lon_step / 2.0
    north = float(latitudes[0]) + lat_step / 2.0
    transform = from_origin(west, north, lon_step, lat_step)
    crs = CRS.from_epsg(4686)
    with rasterio.open(
        nombre_raster, "w", driver="GTiff",
        height=int(grid.shape[0]), width=int(grid.shape[1]),
        count=1, dtype=np.float64, crs=crs, transform=transform,
    ) as dst:
        dst.write(np.asarray(grid, dtype=np.float64), 1)
    print(f"Raster generado     : {nombre_raster}")
    print(f"Filas x columnas    : {grid.shape[0]} x {grid.shape[1]}")
    print(f"Resolucion lat      : {lat_step * 60:.10f} min")
    print(f"Resolucion lon      : {lon_step * 60:.10f} min")


def generacion_raster(ruta_matriz_diseno_pixel,
                      ruta_parametros,
                      raster_puntos_centrales,
                      nombre_raster,
                      param_nom,
                      legacy=False,
                      ruta_puntos=None,
                      col_lat=None, col_lon=None, col_alt=None,
                      ruta_malla=None,
                      col_lat_malla=None, col_lon_malla=None,
                      num_filas=None, para_pixeles=1,
                      filtro="Cup",
                      tam_bloque=2000,
                      bytes_bloque=None,
                      n_lectores=None,
                      chunk_rows=None,
                      usar_gpu=True,
                      skip_benchmark=False):
    del skip_benchmark
    print("=" * 78)
    print("Generador de Raster v2")
    print("=" * 78)

    if not os.path.isfile(ruta_parametros):
        raise FileNotFoundError(ruta_parametros)
    if not os.path.isfile(raster_puntos_centrales):
        raise FileNotFoundError(raster_puntos_centrales)

    t_param = time.perf_counter()
    parametros = np.loadtxt(ruta_parametros, delimiter="\t")
    parametros = np.asarray(parametros, dtype=np.float64).reshape(-1)
    parametros = np.ascontiguousarray(parametros)
    print(f"Parametros leidos   : {parametros.size:,} "
          f"({time.perf_counter() - t_param:.2f}s)")

    if legacy:
        print(f"Modo                : LEGACY (.npy)")
        if ruta_matriz_diseno_pixel is None:
            raise ValueError("Modo legacy requiere --matriz-npy")
        if not os.path.isfile(ruta_matriz_diseno_pixel):
            raise FileNotFoundError(ruta_matriz_diseno_pixel)

        shape, dtype, offset, fortran = leer_header_npy(ruta_matriz_diseno_pixel)
        if len(shape) != 2:
            raise ValueError(f"Se esperaba matriz 2D. shape={shape}")
        if fortran:
            raise ValueError("El .npy esta en orden Fortran; se requiere C.")
        n_filas, n_cols = map(int, shape)
        if parametros.size != n_cols:
            raise ValueError(f"Parametros={parametros.size} != n_cols={n_cols}")
        if detect_gpu() is None:
            raise RuntimeError("Modo legacy requiere GPU disponible.")

        cfg = _auto_config_legacy(
            ruta_matriz_diseno_pixel, n_filas, n_cols,
            bytes_bloque, n_lectores, chunk_rows,
        )
        print(f"Matriz              : {n_filas:,} x {n_cols:,} ({dtype})")
        print(f"Disco               : {cfg['media_type']}")
        print(f"GPU                 : {cfg['gpu']['nombre']}")
        print(f"Filas por bloque    : {cfg['filas_por_bloque']:,}")
        print(f"Bloques             : {cfg['total_bloques']:,}")
        print("=" * 78)

        resultado = np.empty(n_filas, dtype=np.float64)
        t0 = time.perf_counter()
        _generar_legacy(
            ruta_matriz_diseno_pixel, offset, dtype, n_filas, n_cols,
            parametros, cfg["filas_por_bloque"], cfg["total_bloques"],
            cfg["n_lectores"], cfg["host_buffers"], resultado,
        )
        print(f"Calculo terminado   : {(time.perf_counter() - t0)/60:.2f} min")
        print("=" * 78)

    else:
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

        gpu = detect_gpu() if usar_gpu else None
        if gpu is None:
            raise RuntimeError(
                "Modo in-situ requiere GPU (CuPy + CUDA). "
                "Para CPU use modo legacy con --legacy y --matriz-npy."
            )

        print(f"Modo                : IN-SITU con GPU")
        print(f"GPU                 : {gpu['nombre']}")
        print(f"VRAM libre          : {gpu['vram_libre_gb']:.2f} GB")
        print(f"Puntos              : {ruta_puntos}")
        print(f"Malla               : {ruta_malla}")
        print(f"num_filas           : {num_filas}")
        print(f"para_pixeles        : {para_pixeles}")
        print(f"filtro              : {filtro}")
        print(f"tam_bloque          : {tam_bloque:,}")

        t_prep = time.perf_counter()
        prep = preparar_generador(
            ruta_puntos, col_lat, col_lon, col_alt,
            ruta_malla, col_lat_malla, col_lon_malla,
            num_filas, para_pixeles, filtro,
        )
        print(f"n_obs               : {prep['n_obs']:,}")
        print(f"n_malla             : {prep['n_malla']:,}")
        print(f"Preparacion         : {time.perf_counter() - t_prep:.2f}s")

        if parametros.size != prep["n_malla"]:
            raise ValueError(
                f"Parametros={parametros.size} != n_malla={prep['n_malla']}"
            )
        print("=" * 78)

        resultado = np.empty(prep["n_obs"], dtype=np.float64)
        t0 = time.perf_counter()
        _generar_in_situ_gpu(prep, parametros, tam_bloque, resultado)
        print(f"Calculo terminado   : {(time.perf_counter() - t0)/60:.2f} min")
        print("=" * 78)

    t_points = time.perf_counter()
    puntos = _cargar_puntos(raster_puntos_centrales)
    print(f"Puntos leidos       : {len(puntos):,} "
          f"({time.perf_counter() - t_points:.2f}s)")
    grid, latitudes, longitudes = _preparar_raster(
    puntos,
    resultado,
    param_nom,
    raster_puntos_centrales
)
    print(f"Escribiendo GeoTIFF : {nombre_raster}")
    _write_raster(nombre_raster, grid, latitudes, longitudes)

    del puntos, grid, latitudes, longitudes, resultado, parametros
    gc.collect()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Genera un raster (T, Z, Std) en float64."
    )
    parser.add_argument("ruta_parametros", type=str)
    parser.add_argument("raster_puntos_centrales", type=str)
    parser.add_argument("nombre_raster", type=str)
    parser.add_argument("param_nom", type=str, choices=["T", "Z", "Std"])

    parser.add_argument("--legacy", action="store_true", default=False,
                        help="Usa modo legacy leyendo el .npy de la matriz "
                             "de diseno. Por defecto se usa modo in-situ.")

    parser.add_argument("--matriz-npy", "--matriz_npy", type=str, default=None,
                        dest="ruta_matriz_diseno_pixel")
    parser.add_argument("--bytes_bloque", type=int, default=None)
    parser.add_argument("--n_lectores", type=int, default=None)
    parser.add_argument("--chunk_rows", type=int, default=None)

    parser.add_argument("--ruta-puntos", "--ruta_puntos", type=str, default=None,
                        dest="ruta_puntos")
    parser.add_argument("--col-lat", default=None, dest="col_lat")
    parser.add_argument("--col-lon", default=None, dest="col_lon")
    parser.add_argument("--col-alt", default=None, dest="col_alt")
    parser.add_argument("--ruta-malla", "--ruta_malla", type=str, default=None,
                        dest="ruta_malla")
    parser.add_argument("--col-lat-malla", default=None, dest="col_lat_malla")
    parser.add_argument("--col-lon-malla", default=None, dest="col_lon_malla")
    parser.add_argument("--num-filas", "--num_filas", type=int, default=None,
                        dest="num_filas")
    parser.add_argument("--para-pixeles", "--para_pixeles", type=int, default=1,
                        dest="para_pixeles")
    parser.add_argument("--filtro", default="Cup")
    parser.add_argument("--tam-bloque", "--tam_bloque", type=int, default=2000,
                        dest="tam_bloque")

    parser.add_argument("--no_gpu", action="store_true")
    parser.add_argument("--no_benchmark", action="store_true")

    args = parser.parse_args()

    generacion_raster(
        ruta_matriz_diseno_pixel=args.ruta_matriz_diseno_pixel,
        ruta_parametros=args.ruta_parametros,
        raster_puntos_centrales=args.raster_puntos_centrales,
        nombre_raster=args.nombre_raster,
        param_nom=args.param_nom,
        legacy=args.legacy,
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
        bytes_bloque=args.bytes_bloque,
        n_lectores=args.n_lectores,
        chunk_rows=args.chunk_rows,
        usar_gpu=(not args.no_gpu),
        skip_benchmark=args.no_benchmark,
    )