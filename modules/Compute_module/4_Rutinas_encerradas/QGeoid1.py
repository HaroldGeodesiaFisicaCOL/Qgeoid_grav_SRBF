#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
QGeoid1.py

Este script toma la tabla restaurada generada por Restore1.py
y la convierte en un GeoTIFF del QGeoid.

Entrada esperada:
- Un TXT/TSV con columnas al menos:
  'Latitud', 'Longitud' y 'anom_res'
  (si no existe 'anom_res', también acepta 'zeta').

Salida:
- GeoTIFF con la malla del QGeoid.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.crs import CRS
from rasterio.transform import from_origin


def cargar_tabla_restore(ruta_restore_txt: str) -> pd.DataFrame:
    """
    Lee la tabla generada por Restore1.py.
    """
    ruta_restore_txt = Path(ruta_restore_txt)
    if not ruta_restore_txt.exists():
        raise FileNotFoundError(f"No existe el archivo de Restore: {ruta_restore_txt}")

    # Separación flexible: tabulación o espacios múltiples
    try:
        df = pd.read_csv(ruta_restore_txt, sep="\t")
    except Exception:
        df = pd.read_csv(ruta_restore_txt, sep=r"\s+", engine="python")

    return df


def generar_raster_desde_restore(
    df: pd.DataFrame,
    output_filename: str,
    pixel_size_min: float,
    round_decimals: int = 10,
    nodata_val: float = -99999.0,
    crs_epsg: int = 4686,
):
    """
    Genera un GeoTIFF desde la tabla restaurada.
    - Usa Latitud / Longitud como centros de píxel.
    - Usa anom_res como valor a rasterizar.
    - pixel_size_min está en minutos de arco.
    """
    required = {"Latitud", "Longitud"}
    if not required.issubset(df.columns):
        raise ValueError("El dataframe debe contener las columnas 'Latitud' y 'Longitud'.")

    value_col = None
    if "anom_res" in df.columns:
        value_col = "anom_res"
    elif "zeta" in df.columns:
        value_col = "zeta"
    else:
        raise ValueError("El dataframe debe contener 'anom_res' o, en su defecto, 'zeta'.")

    df_local = df.copy()

    # Redondeo para evitar mismatches por precisión de coma flotante
    df_local["Lat_r"] = df_local["Latitud"].astype(float).round(round_decimals)
    df_local["Lon_r"] = df_local["Longitud"].astype(float).round(round_decimals)
    df_local[value_col] = pd.to_numeric(df_local[value_col], errors="coerce")

    # Agrupar por coordenada por si existen duplicados
    df_group = (
        df_local.groupby(["Lat_r", "Lon_r"], as_index=False)[value_col]
        .mean(numeric_only=True)
    )

    latitudes = np.sort(df_group["Lat_r"].unique())[::-1]  # norte a sur
    longitudes = np.sort(df_group["Lon_r"].unique())       # oeste a este

    nrows = len(latitudes)
    ncols = len(longitudes)

    if nrows == 0 or ncols == 0:
        raise ValueError("No hay coordenadas válidas para construir el ráster.")

    pixel_deg = float(pixel_size_min) / 60.0
    if pixel_deg <= 0:
        raise ValueError("pixel_size_min debe ser positivo.")

    raster_data = np.full((nrows, ncols), nodata_val, dtype=np.float64)

    lat_idx = {lat: i for i, lat in enumerate(latitudes)}
    lon_idx = {lon: j for j, lon in enumerate(longitudes)}

    for _, row in df_group.iterrows():
        i = lat_idx.get(row["Lat_r"])
        j = lon_idx.get(row["Lon_r"])
        if i is None or j is None:
            continue

        val = row[value_col]
        if pd.isna(val):
            raster_data[i, j] = nodata_val
        else:
            raster_data[i, j] = np.float64(val)

    half = pixel_deg / 2.0
    west = float(longitudes.min()) - half
    north = float(latitudes.max()) + half
    transform = from_origin(west, north, pixel_deg, pixel_deg)

    crs = CRS.from_epsg(int(crs_epsg))

    profile = {
        "driver": "GTiff",
        "height": nrows,
        "width": ncols,
        "count": 1,
        "dtype": "float64",
        "crs": crs,
        "transform": transform,
        "nodata": nodata_val,
        "compress": "lzw",
    }

    output_filename = Path(output_filename)
    output_filename.parent.mkdir(parents=True, exist_ok=True)

    with rasterio.open(output_filename, "w", **profile) as dst:
        dst.write(raster_data, 1)

    print(f"[INFO] GeoTIFF escrito: {output_filename}")
    print(f"[INFO] CRS: {crs}")
    print(f"[INFO] Transform: {transform}")
    print(f"[INFO] Bounds: {west}, {north}")


def main(ruta_restore_txt: str, salida_tif: str, pixel_size_min: float, crs_epsg: int):
    df = cargar_tabla_restore(ruta_restore_txt)
    generar_raster_desde_restore(
        df=df,
        output_filename=salida_tif,
        pixel_size_min=pixel_size_min,
        crs_epsg=crs_epsg,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generador de QGeoid desde la tabla Restore")
    parser.add_argument("ruta_restore_txt", help="TXT/TSV producido por Restore1.py")
    parser.add_argument("salida_tif", help="GeoTIFF de salida para el QGeoid")
    parser.add_argument("pixel_size", type=float, help="Tamaño de píxel en minutos de arco")
    parser.add_argument("--crs_epsg", type=int, default=4686, help="EPSG del sistema de referencia (default: 4686)")

    args = parser.parse_args()

    main(
        ruta_restore_txt=args.ruta_restore_txt,
        salida_tif=args.salida_tif,
        pixel_size_min=args.pixel_size,
        crs_epsg=args.crs_epsg,
    )