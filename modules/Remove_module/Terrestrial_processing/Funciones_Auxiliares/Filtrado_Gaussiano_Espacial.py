import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from pyproj import Geod


def ordenar_linea_por_distancia(lat, lon, ellps="GRS80"):
    """
    Ordena los puntos de una línea según la distancia acumulada
    a lo largo de la trayectoria de vuelo.
    """
    geod = Geod(ellps=ellps)

    _, _, dist = geod.inv(
        lon[:-1], lat[:-1],
        lon[1:],  lat[1:]
    )

    s = np.concatenate(([0.0], np.cumsum(dist)))
    return np.argsort(s)


def filtro_gaussiano_espacial(
    df_path,
    col_lat="Latitud",
    col_lon="Longitud",
    col_alt="Altura",
    col_grav="Pert_Grav",
    col_linea="Numero_Linea",
    col_proyecto="Proyecto",
    ellps="GRS80",
    sigma_max_samples=20,
    umbral_mgal=20.0   # ← CONDICIONANTE DE SEGURIDAD
):

    df = pd.read_csv(df_path, sep=',')
    geod = Geod(ellps=ellps)

    resultados = []
    reporte = []

    for proyecto, df_proj in df.groupby(col_proyecto):

        # --- calcular espaciamientos por línea ---
        dx_all = []
        alturas = []

        for linea, datos in df_proj.groupby(col_linea):

            lat = datos[col_lat].values
            lon = datos[col_lon].values

            if len(datos) < 2:
                continue

            _, _, dist = geod.inv(
                lon[:-1], lat[:-1],
                lon[1:],  lat[1:]
            )

            dx_all.extend(dist)
            alturas.extend(datos[col_alt].values)

        dx_all = np.asarray(dx_all)
        alturas = np.asarray(alturas)

        dx_med = np.median(dx_all)
        dx_p25 = np.percentile(dx_all, 25)
        dx_p75 = np.percentile(dx_all, 75)

        h_med = np.median(alturas)

        # --- sigma físico para RESIDUALES ---
        sigma_km = 0.15 * h_med / 1000.0
        sigma_samples = (sigma_km * 1000.0) / dx_med
        sigma_samples_eff = min(sigma_samples, sigma_max_samples)

        # --- filtrado correcto por línea ---
        for linea, datos in df_proj.groupby(col_linea):

            lat = datos[col_lat].values
            lon = datos[col_lon].values
            g = datos[col_grav].values

            if len(datos) < 5:
                datos = datos.copy()
                datos["Pert_Grav_Filt"] = g
                datos["Residuo_HF"] = 0.0
                resultados.append(datos)
                continue

            # 1) ordenar por trayectoria real
            idx = ordenar_linea_por_distancia(lat, lon, ellps=ellps)
            g_ord = g[idx]

            # 2) eliminar tendencia lineal (NO media)
            x = np.arange(len(g_ord))
            coef = np.polyfit(x, g_ord, 1)
            tendencia = np.polyval(coef, x)
            g_detr = g_ord - tendencia

            # 3) filtrar solo la señal fluctuante
            g_filt_detr = gaussian_filter1d(
                g_detr,
                sigma=sigma_samples_eff,
                mode="nearest"
            )

            # 4) reconstrucción física (sin offsets)
            g_filt_ord = g_filt_detr + tendencia

            # 5) volver al orden original
            g_filt = np.empty_like(g)
            g_filt[idx] = g_filt_ord

            # --- CONDICIONANTE DE SEGURIDAD ---
            delta = np.abs(g_filt - g)
            mask_exceso = delta > umbral_mgal

            # donde el filtro introduce cambios no físicos → conservar original
            g_filt[mask_exceso] = g[mask_exceso]

            datos = datos.copy()
            datos["Pert_Grav_Filt"] = g_filt
            datos["Residuo_HF"] = g - g_filt

            resultados.append(datos)

        # --- reporte por proyecto ---
        reporte.append({
            "Proyecto": proyecto,
            "dx_med_m": dx_med,
            "dx_p25_m": dx_p25,
            "dx_p75_m": dx_p75,
            "altura_med_m": h_med,
            "sigma_km": sigma_km,
            "sigma_samples_teorico": sigma_samples,
            "sigma_samples_usado": sigma_samples_eff
        })

    return (
        pd.concat(resultados, ignore_index=True),
        pd.DataFrame(reporte)
    )
