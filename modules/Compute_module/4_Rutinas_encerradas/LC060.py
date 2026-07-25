#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LC_lcurve.py — L-curve con muestreo grueso y fino en escala log-log
Versión: igual interfaz, selección de lambda por curvatura Hansen-style
(método de spline cúbica sobre t = log(lambda)), y gráficos comparativos
entre curvatura discreta y curvatura Hansen (nodos + malla densa).

Uso:
 python LC_lcurve.py --listado listado.txt --pesos pesos.csv --normas normas.txt \
     --out_params beta.txt --out_cov cov.npy --out_fig resumen_lambda.pdf
"""
import argparse
import logging
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from numpy.linalg import solve
from tqdm import tqdm
from scipy.interpolate import CubicSpline

# ----------------------------
# Configuración de LaTeX en matplotlib
# ----------------------------
USE_TEX = False
try:
    if matplotlib.checkdep_usetex(True):
        USE_TEX = True
except Exception:
    USE_TEX = False

plt.rcParams['text.usetex'] = USE_TEX
plt.rcParams['font.family'] = 'serif'

# ----------------------------
# Logging
# ----------------------------
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s: %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

# ----------------------------
# I/O helpers
# ----------------------------
def load_X_matrices(paths):
    mats = []
    for p in paths:
        mats.append(np.load(p))
    return mats

def load_y_vectors(paths, colname='Perturbaciones_residuales', scale=1e5):
    y_list = []
    for p in paths:
        df = pd.read_csv(p, sep='\t', engine='python')
        if colname not in df.columns:
            raise ValueError(f"Columna '{colname}' no encontrada en {p}. Columnas: {df.columns.tolist()}")
        y = df[colname].values.reshape(-1, 1) / scale
        y_list.append(y)
    return y_list

# ----------------------------
# Construcción de la parte independiente de lambda
# ----------------------------
def precompute_base_system(X_list, y_list, sigma2_list):
    """
    Calcula una sola vez la parte que no depende de lambda:

        M = sum_i (sigma1 / s_i^2) X_i^T X_i
        b = sum_i (sigma1 / s_i^2) X_i^T y_i

    donde sigma1 = sigma2_list[0].

    Luego, para cada lambda:
        N(lambda) = M + lambda * I
        N(lambda) beta = b
    """
    n = X_list[0].shape[1]
    sigma1 = float(sigma2_list[0])

    M = np.zeros((n, n))
    rhs = np.zeros((n, 1))

    for X, s2 in zip(X_list, sigma2_list):
        M += (sigma1 / float(s2)) * (X.T @ X)

    for X, s2, y in zip(X_list, sigma2_list, y_list):
        rhs += (sigma1 / float(s2)) * (X.T @ y)

    return sigma1, M, rhs

def build_N_from_base(M_base, lambda_L):
    n = M_base.shape[0]
    return M_base + lambda_L * np.eye(n)

# ----------------------------
# Normas (igual que antes — no ponderada para mantener compatibilidad)
# ----------------------------
def compute_residual_norm(X_list, y_list, beta):
    res_parts = [X @ beta - y for X, y in zip(X_list, y_list)]
    res = np.vstack(res_parts)
    return np.linalg.norm(res)

def compute_solution_norm(beta):
    return np.linalg.norm(beta)

# ----------------------------
# Curvatura discreta (tu función original, fallback)
# ----------------------------
def calculate_curvature_log(pts):
    pts = np.asarray(pts, dtype=float)
    m = len(pts)
    if m < 3:
        return np.zeros(m)
    kappa = np.zeros(m)
    for i in range(1, m-1):
        p1, p2, p3 = pts[i-1], pts[i], pts[i+1]
        a = np.linalg.norm(p2 - p1)
        b = np.linalg.norm(p3 - p2)
        c = np.linalg.norm(p3 - p1)
        s = 0.5 * (a + b + c)
        area_term = s*(s-a)*(s-b)*(s-c)
        if area_term > 0:
            area = np.sqrt(area_term)
            denom = (a * b * c)
            if denom > 0:
                kappa[i] = 4.0 * area / denom
    return kappa

# ----------------------------
# Curvatura Hansen: spline en t = log(lambda)
# ----------------------------
def _curvature_from_spline_on_t(t_eval, cs_x, cs_y):
    # cs_x and cs_y are CubicSpline objects; we use derivative evaluation:
    x1 = cs_x(t_eval, 1)
    x2 = cs_x(t_eval, 2)
    y1 = cs_y(t_eval, 1)
    y2 = cs_y(t_eval, 2)
    denom = (x1 * x1 + y1 * y1)
    denom_p = np.power(denom, 1.5)
    with np.errstate(divide='ignore', invalid='ignore'):
        k = np.abs(x1 * y2 - y1 * x2) / (denom_p + 1e-300)
    k = np.nan_to_num(k, nan=0.0, posinf=0.0, neginf=0.0)
    return k

def calculate_curvature_log_hansen(lambdas, rn, sn, dense_factor=10, n_eval_min=400):
    """
    Ajusta splines cúbicas en t = log(lambda) para x(t)=log(rn), y(t)=log(sn),
    evalúa la curvatura analítica en una malla densa y devuelve:
      - k_nodes: curvatura en los nodos originales (len = len(lambdas))
      - t_dense, k_dense: malla densa y curvatura correspondiente (para graficar)
      - idx_dense_max: índice en t_dense del máximo de curvatura
      - idx_node_closest: índice del nodo cuyo t está más cerca del t_dense del máximo
    """
    lambdas = np.asarray(lambdas, dtype=float)
    rn = np.asarray(rn, dtype=float)
    sn = np.asarray(sn, dtype=float)

    # defensiva: evitar ceros o negativos para log
    eps = 1e-300
    rn = np.maximum(rn, eps)
    sn = np.maximum(sn, eps)

    m = lambdas.size
    if m < 4:
        # fallback a discreta
        pts_log = np.column_stack((np.log(rn), np.log(sn)))
        return calculate_curvature_log(pts_log), None, None, None

    t = np.log(lambdas)
    x = np.log(rn)
    y = np.log(sn)

    # asegurar t estrictamente creciente
    if not np.all(np.diff(t) > 0):
        eps_j = 1e-12
        for i in range(1, len(t)):
            if t[i] <= t[i-1]:
                t[i] = t[i-1] + eps_j
                eps_j *= 10.0

    cs_x = CubicSpline(t, x, extrapolate=True)
    cs_y = CubicSpline(t, y, extrapolate=True)

    n_eval = max(n_eval_min, int(len(t) * dense_factor))
    t_dense = np.linspace(t[0], t[-1], n_eval)
    k_dense = _curvature_from_spline_on_t(t_dense, cs_x, cs_y)

    # máximo en la malla densa
    idx_dense_max = int(np.argmax(k_dense))
    t_opt_dense = t_dense[idx_dense_max]

    # curvatura en nodos
    k_nodes = _curvature_from_spline_on_t(t, cs_x, cs_y)

    # índice de nodo más cercano al máximo denso
    idx_node_closest = int(np.argmin(np.abs(t - t_opt_dense)))

    return k_nodes, (t_dense, k_dense), idx_dense_max, idx_node_closest

# ----------------------------
# Selección de lambda: devuelve curvaturas para comparar
# ----------------------------
def pick_optimal_lambda(lambdas, rn, sn):
    lambdas = np.asarray(lambdas, dtype=float)
    rn = np.asarray(rn, dtype=float).ravel()
    sn = np.asarray(sn, dtype=float).ravel()

    # calculo curvatura discreta (fallback / comparación)
    pts_log = np.column_stack((np.log(np.maximum(rn, 1e-300)), np.log(np.maximum(sn, 1e-300))))
    curv_discrete = calculate_curvature_log(pts_log)
    if curv_discrete.size >= 4:
        curv_discrete[:2] = 0.0
        curv_discrete[-2:] = 0.0

    # calculo curvatura Hansen (spline)
    try:
        k_nodes, dense_info, idx_dense_max, idx_node_closest = calculate_curvature_log_hansen(lambdas, rn, sn)
        # dense_info puede ser None si fallback
        if dense_info is None:
            # fallback: usar discreta
            idx = int(np.nanargmax(curv_discrete))
            return idx, float(lambdas[idx]), curv_discrete, None, None, None
        t_dense, k_dense = dense_info
        # aprovechar idx_node_closest (índice del nodo más cercano al máximo denso)
        idx = int(idx_node_closest)
        # seguridad en límites
        idx = max(0, min(len(lambdas)-1, idx))
        return idx, float(lambdas[idx]), curv_discrete, k_nodes, (t_dense, k_dense), idx_dense_max
    except Exception as e:
        logger.warning(f"Fallo en calculate_curvature_log_hansen: {e}. Fallback a discreta.")
        idx = int(np.nanargmax(curv_discrete))
        return idx, float(lambdas[idx]), curv_discrete, None, None, None

# ----------------------------
# Main
# ----------------------------
def main():
    parser = argparse.ArgumentParser(description="L-curve: muestreo grueso y fino (Hansen-style lambda)")
    parser.add_argument('--listado',    required=True, help='.txt con rutas: .npy (X) y .txt/.csv (y) por fila, sep tab')
    parser.add_argument('--pesos',      required=True, help='CSV/TSV de pesos (filas con varianzas y un último valor)')
    parser.add_argument('--normas',     required=True, help='TXT/CSV con normas [res, sol] por fila')
    parser.add_argument('--out_params', required=True, help='Salida parámetros optimizados (txt)')
    parser.add_argument('--out_cov',    required=True, help='Salida matriz covarianza (.npy)')
    parser.add_argument('--out_residuals', required=True, help='Salida residuos finales e = y - Aβ')
    parser.add_argument('--out_fig',    default='Resumen_Curvas_Lambda.pdf', help='PDF resumen')
    args = parser.parse_args()

    # 1) Leer listado
    info = pd.read_csv(args.listado, sep='\t', header=None, engine='python')
    if info.shape[1] < 2:
        raise ValueError('El archivo listado debe tener al menos 2 columnas: ruta_X \\t ruta_y por fila')
    X_paths = info.iloc[:, 0].tolist()
    y_paths = info.iloc[:, 1].tolist()

    X_list = load_X_matrices(X_paths)
    y_list = load_y_vectors(y_paths)

    # 2) Pesos y normas externas
    pesos = pd.read_csv(args.pesos, header=None, engine='python').values
    normas = np.loadtxt(args.normas)
    if normas.ndim == 1:
        normas = normas.reshape(-1, 2)

    # 3) lambda inicial desde normas externas si aplica
    if normas.shape[0] < 3:
        logger.warning('Se requieren al menos 3 pares de normas externas para calcular curvatura externa. Se usará lambda inicial simple.')
        lam0 = 1.0
    else:
        pts_ext = np.column_stack((np.log(np.maximum(normas[:, 0], 1e-300)), np.log(np.maximum(normas[:, 1], 1e-300))))
        curv_ext = calculate_curvature_log(pts_ext)
        if curv_ext.size >= 4:
            curv_ext[:2] = 0.0
            curv_ext[-2:] = 0.0
        idx_ext = int(np.nanargmax(curv_ext))
        if idx_ext >= pesos.shape[0]:
            idx_ext = np.argmin(pesos[:, 0])
        lam0 = float(pesos[idx_ext, 0]) / float(pesos[idx_ext, -1])
        logger.info(f"Lambda inicial (desde normas externas): {lam0} (idx {idx_ext})")

    # 4) fila de pesos (misma convención tuya)
    idx_min_peso = pesos.shape[0] - 1
    sigma_flat = pesos[idx_min_peso, :-1]
    sigma1 = float(sigma_flat[0])
    logger.info(f"Fila con menor primer valor de peso: {idx_min_peso} (peso={pesos[idx_min_peso, 0]})")

    # 4.1) Precomputar una sola vez la parte independiente de lambda
    sigma1_pre, M_base, rhs_base = precompute_base_system(X_list, y_list, sigma_flat)

    # Por coherencia con tu formulación original, sigma1_pre debe coincidir con sigma1
    # (se conserva el valor original para el guardado de la covarianza).
    if not np.isclose(sigma1_pre, sigma1):
        logger.warning("sigma1 calculado en precompute_base_system no coincide exactamente con sigma1 leído desde pesos.")

    # 5) coarse sweep
    lambdas_coarse = lam0 * np.logspace(-4, 4, 20)
    rn_c, sn_c = [], []
    for lam in tqdm(lambdas_coarse, desc='Coarse sweep'):
        N = build_N_from_base(M_base, lam)
        beta = solve(N, rhs_base)
        rn_c.append(compute_residual_norm(X_list, y_list, beta))
        sn_c.append(compute_solution_norm(beta))

    # compute curvatures and choose coarse best (Hansen)
    idx_c, lam_c, curv_grues_disc, curv_grues_h_nodes, dense_grues, idx_dense_grues = pick_optimal_lambda(lambdas_coarse, rn_c, sn_c)
    logger.info(f"Coarse best idx={idx_c}, lambda_coarse={lam_c:.6g}")

    # 6) fine sweep around coarse best
    low = float(lambdas_coarse[max(idx_c-1, 0)])
    high = float(lambdas_coarse[min(idx_c+1, len(lambdas_coarse)-1)])
    if low == high:
        lambdas_fine = np.array([low])
    else:
        lambdas_fine = np.logspace(np.log10(low), np.log10(high), 20)

    rn_f, sn_f = [], []
    for lam in tqdm(lambdas_fine, desc='Fine sweep'):
        N = build_N_from_base(M_base, lam)
        beta = solve(N, rhs_base)
        rn_f.append(compute_residual_norm(X_list, y_list, beta))
        sn_f.append(compute_solution_norm(beta))

    # compute curvatures and choose optimal lambda using Hansen approach
    idx_opt, lam_opt, curv_fine_disc, curv_fine_h_nodes, dense_fine, idx_dense_fine = pick_optimal_lambda(lambdas_fine, rn_f, sn_f)
    logger.info(f"Lambda óptimo (Hansen-style): {lam_opt} (idx {idx_opt})")

    # For plotting convenience unpack dense info
    t_dense_f, k_dense_f = (None, None)
    if dense_fine is not None:
        t_dense_f, k_dense_f = dense_fine

    t_dense_c, k_dense_c = (None, None)
    if dense_grues is not None:
        t_dense_c, k_dense_c = dense_grues

    # 8) Guardar resultado final
    N_opt = build_N_from_base(M_base, lam_opt)

    cov_opt = np.linalg.inv(N_opt)

    beta_opt = cov_opt @ rhs_base

    # ----------------------------------
    # Residuos finales
    # e = y - A beta
    # ----------------------------------
    residuals_list = []

    for X, y in zip(X_list, y_list):

        e = y - (X @ beta_opt)

        residuals_list.append(e)

    residuals = np.vstack(residuals_list)

    # ----------------------------------
    # Guardado
    # ----------------------------------
    np.savetxt(
        args.out_params,
        beta_opt.reshape(-1, 1),
        fmt="%.16e"
    )

    np.save(
        args.out_cov,
        sigma1 * cov_opt
    )

    np.savetxt(
        args.out_residuals,
        residuals,
        fmt="%.16e"
    )

    logger.info(
        f"Guardado beta en {args.out_params}, "
        f"cov en {args.out_cov} y "
        f"residuos en {args.out_residuals}"
    )

    # 9) Graficar resumen y comparaciones
    try:
        fig, axs = plt.subplots(3, 2, figsize=(16, 15))

        # A) Normas externas (log-log) - si existen
        if 'curv_ext' in locals():
            axs[0,0].plot(np.log(np.maximum(normas[:,0], 1e-300)), np.log(np.maximum(normas[:,1], 1e-300)), 'bo-')
            axs[0,0].axvline(np.log(np.maximum(normas[idx_ext,0], 1e-300)), color='red', linestyle='--')
        else:
            axs[0,0].text(0.5, 0.5, 'No hay normas externas suficientes', ha='center')
        axs[0,0].set_title('Normas externas (log-log)')
        axs[0,0].set_xlabel(r'$\log\|A\hat{\beta}_{\lambda}-y\|$')
        axs[0,0].set_ylabel(r'$\log\|\hat{\beta}_{\lambda}-\mu\|$')
        axs[0,0].grid(True)

        # B) Curvatura normas externas (discreta)
        axs[0,1].plot(curv_ext if 'curv_ext' in locals() else [0], 'b.-')
        if 'idx_ext' in locals():
            axs[0,1].axvline(idx_ext, color='red', linestyle='--')
        axs[0,1].set_title('Curvatura normas externas (discreta)')
        axs[0,1].set_xlabel('Índice')
        axs[0,1].set_ylabel('Curvatura')
        axs[0,1].grid(True)

        # C) L-curve coarse & fine (log-log)
        axs[1,0].loglog(rn_c, sn_c, 'c.--', label='Coarse (puntos)')
        axs[1,0].loglog(rn_f, sn_f, 'k.-', label='Fine (puntos)')
        # mark Hansen-chosen lambda (node)
        axs[1,0].scatter([rn_f[idx_opt]], [sn_f[idx_opt]], marker='o', s=100, facecolors='none', edgecolors='red', label=rf'$\lambda^*_{{H}} = {lam_opt:.3g}$')
        axs[1,0].set_title('L-curve: coarse & fine')
        axs[1,0].set_xlabel(r'$\|A\hat{\beta}_{\lambda}-y\|$ (escala log)')
        axs[1,0].set_ylabel(r'$\|\hat{\beta}_{\lambda}-\mu\|$ (escala log)')
        axs[1,0].legend()
        axs[1,0].grid(True, which='both')

        # D) Curvatura coarse: discreta vs Hansen (nodos)
        axs[1,1].plot(curv_grues_disc, 'g.-', label='Discreta (coarse)')
        if curv_grues_h_nodes is not None:
            axs[1,1].plot(curv_grues_h_nodes, 'b.-', label='Hansen (nodos, coarse)')
        axs[1,1].axvline(idx_c, color='red', linestyle='--', label='idx coarse (discreta)')
        if curv_grues_h_nodes is not None:
            axs[1,1].axvline(idx_c, color='red', linestyle='--')
        axs[1,1].set_title('Curvatura muestreo grueso')
        axs[1,1].set_xlabel('Índice')
        axs[1,1].set_ylabel('Curvatura')
        axs[1,1].legend()
        axs[1,1].grid(True)

        # E) L-curve detalle (fino) con ambos marcadores (discreta/Hansen)
        axs[2,0].loglog(rn_f, sn_f, 'k.-', label='Fine (puntos)')
        axs[2,0].scatter([rn_f[idx_opt]], [sn_f[idx_opt]], marker='o', s=100, facecolors='none', edgecolors='red', label=rf'$\lambda^*_{{H}} = {lam_opt:.3g}$')
        # also mark discrete-curvature best (if different)
        idx_opt_discrete = int(np.nanargmax(curv_fine_disc))
        if idx_opt_discrete != idx_opt:
            axs[2,0].scatter([rn_f[idx_opt_discrete]], [sn_f[idx_opt_discrete]], marker='s', s=100, facecolors='none', edgecolors='orange', label=rf'$\lambda^*_{{disc}} = {lambdas_fine[idx_opt_discrete]:.3g}$')
        axs[2,0].set_title('L-curve: detalle (fino)')
        axs[2,0].set_xlabel(r'$\|A\hat{\beta}_{\lambda}-y\|$ (escala log)')
        axs[2,0].set_ylabel(r'$\|\hat{\beta}_{\lambda}-\mu\|$ (escala log)')
        axs[2,0].legend()
        axs[2,0].grid(True, which='both')

        # F) Curvatura muestreo fino: discreta vs Hansen (nodos + densa)
        axs[2,1].plot(curv_fine_disc, 'g.-', label='Discreta (fino)')
        if curv_fine_h_nodes is not None:
            axs[2,1].plot(curv_fine_h_nodes, 'b.-', label='Hansen (nodos, fino)')
        # mark node-based optimum
        axs[2,1].axvline(idx_opt, color='red', linestyle='--', label='Hansen: idx nodo más cercano')
        # if dense data exists, plot and mark dense maximum
        if k_dense_f is not None:
            # map dense t to index-like x for plotting convenience: use t axis (log lambda)
            # we will create an inset-like additional figure for dense curve below.
            axs[2,1].plot([], [])  # no-op but keep legend consistent
        axs[2,1].set_title('Curvatura muestreo fino (comparación)')
        axs[2,1].set_xlabel('Índice')
        axs[2,1].set_ylabel('Curvatura')
        axs[2,1].legend()
        axs[2,1].grid(True)

        plt.tight_layout()
        plt.savefig(args.out_fig)
        logger.info(f"Figura resumen guardada en: {args.out_fig}")

        # --- figuras adicionales: curvatura densa para fine (spline)
        if k_dense_f is not None:
            fig2 = plt.figure(figsize=(9,6))
            plt.plot(t_dense_f, k_dense_f, 'b-', label='Curvatura densa (spline)')
            # mark dense maximum
            idx_dense = int(np.argmax(k_dense_f))
            plt.axvline(t_dense_f[idx_dense], color='red', linestyle='--', label='Máx curvatura densa')
            # mark node positions (log lambdas)
            plt.scatter(np.log(lambdas_fine), np.interp(np.log(lambdas_fine), t_dense_f, k_dense_f), marker='o', color='gray', s=30, alpha=0.5, label='Nodos (log λ)')
            plt.title('Curvatura densa (Hansen) — muestreo fino')
            plt.xlabel('t = log(lambda)')
            plt.ylabel('Curvatura')
            plt.legend()
            plt.grid(True)
            fname2 = args.out_fig.replace('.pdf', '_dense_fine.pdf')
            plt.savefig(fname2)
            logger.info(f"Figura curvatura densa guardada en: {fname2}")

            # comparison nodes vs dense (same t axis)
            fig3 = plt.figure(figsize=(9,6))
            plt.plot(np.log(lambdas_fine), curv_fine_disc, 'go-', label='Curvatura discreta (nodos)')
            plt.plot(t_dense_f, k_dense_f, 'b-', alpha=0.8, label='Curvatura densa (spline)')
            plt.scatter(np.log(lambdas_fine[idx_opt]), curv_fine_disc[idx_opt], color='red', s=80, label='λ* Hansen (nodo)')
            plt.title('Comparación: curvatura nodos vs curvatura densa (fino)')
            plt.xlabel('t = log(lambda)')
            plt.ylabel('Curvatura')
            plt.legend()
            plt.grid(True)
            fname3 = args.out_fig.replace('.pdf', '_compare_fine.pdf')
            plt.savefig(fname3)
            logger.info(f"Figura comparación nodos/densa guardada en: {fname3}")

    except Exception as e:
        logger.warning(f"No pude generar figura(s) resumen/comparativa: {e}")

    plt.show()

if __name__ == '__main__':
    main()