import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from scipy import stats
from pyproj import Transformer
import logging

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s: %(message)s', datefmt='%H:%M:%S')

def detect_outliers(
    df,
    lat_col,
    lon_col,
    val_col,
    k_neighbors=2800,
    max_radius_km=100,
    p_idw=1,
    local_multiplier=8,
    pct_epsilon=99,
    n_iter=1,
    crs_from="EPSG:4686",
    crs_to="EPSG:9377",
    save_prefix="resultado"
):
    df = df.copy().reset_index(drop=True)

    # Proyección a metros
    transformer = Transformer.from_crs(crs_from, crs_to, always_xy=True)
    Xp, Yp = transformer.transform(df[lon_col].values, df[lat_col].values)
    df["X_PROJ"], df["Y_PROJ"] = Xp, Yp

    coords = df[["X_PROJ", "Y_PROJ"]].to_numpy()
    data = df[val_col].to_numpy()
    n = len(df)

    epsilon_global = float(np.percentile(np.abs(data), pct_epsilon))
    logging.info(f"Epsilon global (P{pct_epsilon}): {epsilon_global:.6f} mGal")

    # Inicial: candidatos NMAD global
    med = np.median(data)
    mad = np.median(np.abs(data - med))
    nmad = 1.4826 * mad
    candidatos = np.where((data > med + 3 * nmad) | (data < med - 3 * nmad))[0]
    logging.info(f"Candidatos iniciales NMAD: {len(candidatos)} / {n}")

    outlier_mask = np.zeros(n, dtype=bool)
    interp = data.copy()

    for it in range(n_iter):
        mask_no = ~(np.isin(np.arange(n), candidatos) | outlier_mask)
        coords_no = coords[mask_no]
        data_no = data[mask_no]
        tree = cKDTree(coords_no)

        for idx in candidatos:
            pt = coords[idx]
            dists, idxs = tree.query(pt, k=k_neighbors, p=2)
            if np.isscalar(dists):
                dists = np.array([dists])
                idxs = np.array([idxs])

            valid_mask = dists <= max_radius_km * 1000
            if not np.any(valid_mask):
                interp[idx] = data[idx]  # sin vecinos
                continue
            sel_dists = dists[valid_mask]
            sel_vals = data_no[idxs[valid_mask]]

            if np.any(sel_dists == 0.0):
                interp[idx] = sel_vals[sel_dists == 0.0][0]
            else:
                w = 1.0 / (sel_dists**p_idw)
                interp[idx] = np.dot(sel_vals, w) / w.sum()

        # Umbral local para cada candidato
        local_thresh = np.full(n, np.inf)
        for idx in candidatos:
            vecinos = tree.query_ball_point(coords[idx], max_radius_km * 1000)
            if not vecinos:
                continue
            neigh_vals = data_no[vecinos]
            med_loc = np.median(neigh_vals)
            mad_loc = np.median(np.abs(neigh_vals - med_loc))
            nmad_loc = 1.4826 * mad_loc
            local_thresh[idx] = local_multiplier * nmad_loc

        difs = np.abs(data - interp)
        confirmar = ((difs > epsilon_global) & (np.abs(data) > epsilon_global)) | (difs > local_thresh)
        nuevos_outliers = np.where(confirmar & ~outlier_mask)[0]
        outlier_mask[nuevos_outliers] = True
        logging.info(f"Iter {it+1}: nuevos outliers {len(nuevos_outliers)}")

        # Recalcular NMAD sin outliers
        med = np.median(data[~outlier_mask])
        mad = np.median(np.abs(data[~outlier_mask] - med))
        nmad = 1.4826 * mad
        candidatos = np.where((data > med + 3 * nmad) | (data < med - 3 * nmad))[0]

    # Resultado final
    df["interpolacion"] = interp
    df["diferencia_abs"] = np.abs(data - interp)
    df["es_outlier"] = outlier_mask
    df["es_candidato_NMAD"] = False
    df.loc[candidatos, "es_candidato_NMAD"] = True


    # Imprimir max/min de datos limpios
    limpios = df.loc[~df.es_outlier, val_col]
    print("\n--- Estadísticas datos limpios ---")
    print(f"Min: {limpios.min():.4f}")
    print(f"Max: {limpios.max():.4f}")
    print(f"Media: {limpios.mean():.4f}")
    print(f"Desv.Est.: {limpios.std():.4f}")
    print(f"N puntos limpios: {len(limpios)}")
    print(f"N outliers: {outlier_mask.sum()}")

    # Dispersión espacial
    plt.figure(figsize=(8,6))
    plt.scatter(df.loc[~df.es_outlier, "X_PROJ"], df.loc[~df.es_outlier, "Y_PROJ"], s=1, label="Limpios")
    plt.scatter(df.loc[df.es_outlier, "X_PROJ"], df.loc[df.es_outlier, "Y_PROJ"], s=2, c='r', label="Outliers")
    plt.legend()
    plt.title("Dispersión espacial")
    plt.xlabel("X (m)")
    plt.ylabel("Y (m)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{save_prefix}_spatial.pdf", dpi=600)
    
    return df.loc[~df.es_outlier],df.loc[df.es_outlier],epsilon_global



