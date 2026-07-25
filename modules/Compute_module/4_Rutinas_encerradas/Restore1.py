#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Restore1.py  (versión reestructurada para pipeline robusto)

Este script:
1) NO ejecuta GravLab.
2) Usa las expansiones ya generadas por GravLab_expand.py.
3) Construye el dataframe restaurado con Restas_principal.
4) Interpola el grid ERTM.
5) Calcula iterativamente el restore.
6) Guarda la tabla final con la columna anom_res.

Entrada esperada (compatible con el pipeline robusto):
- archivo_puntos           -> archivo principal de puntos (pixeles_txt o equivalente)
- expansion_ggm            -> expansión GGM
- expansion_topo1          -> expansión topográfica 0_719
- expansion_topo2          -> expansión topográfica 0_2159
- grid_ertm                -> grid ERTM
- salida_restore           -> archivo final Restore_f{freq}.txt
- tolerancia               -> tolerancia numérica
- max_iter                 -> iteraciones máximas
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import pygmt

from Funciones_Auxiliares.Restas_Remove import Restas_principal


# ---------------------------------------------------------------------
# Constantes geodésicas
# ---------------------------------------------------------------------
gamma_a = 978032.67715
gamma_b = 983218.63685
f = 0.00335281068118
m = 0.00344978600308
a = 6378137.0
b = 6356752.3141


# ---------------------------------------------------------------------
# Utilidades
# ---------------------------------------------------------------------
def asegurar_txt(path) -> str:
    """
    Si el archivo no tiene extensión .txt, la agrega.
    Si el archivo sin extensión no existe pero el .txt sí, usa el .txt.
    """
    p = Path(path)
    if p.exists():
        return str(p)

    if p.suffix.lower() != ".txt":
        p_txt = p.with_suffix(".txt")
        if p_txt.exists():
            return str(p_txt)
        return str(p_txt)

    return str(p)


def interpolado_modelos(df: pd.DataFrame, grid_path: str, nombre_columna: str) -> pd.DataFrame:
    """
    Interpola valores desde un grid sobre puntos dados por Longitud/Latitud.
    """
    if "Longitud" not in df.columns or "Latitud" not in df.columns:
        raise ValueError("El dataframe debe contener las columnas 'Latitud' y 'Longitud'.")

    coords = list(zip(df["Longitud"], df["Latitud"]))
    valores = np.asarray(
        pygmt.grdtrack(points=coords, grid=grid_path, interpolation="l")
    )[:, 2]

    df[nombre_columna] = np.nan_to_num(valores, nan=0.0)
    return df


# ---------------------------------------------------------------------
# Gravedad normal
# ---------------------------------------------------------------------
def gravedad_teorica_somigliana(lat):
    lat = np.asarray(lat, dtype=float)
    lat_rad = np.radians(lat)

    num = a * gamma_a * np.cos(lat_rad) ** 2 + b * gamma_b * np.sin(lat_rad) ** 2
    den = np.sqrt(a**2 * np.cos(lat_rad) ** 2 + b**2 * np.sin(lat_rad) ** 2)

    return num / den


def gamma_h(lat, gamma_0, h):
    lat = np.asarray(lat, dtype=float)
    gamma_0 = np.asarray(gamma_0, dtype=float)
    h = np.asarray(h, dtype=float)

    lat_rad = np.radians(lat)
    term1 = 1 - ((2 / a) * (1 + f + m - 2 * f * np.sin(lat_rad) ** 2) * h)
    term2 = (3 / a**2) * h**2

    return gamma_0 * (term1 + term2)


# ---------------------------------------------------------------------
# Construcción del dataframe restaurado
# ---------------------------------------------------------------------
def cargar_datos(
    archivo_puntos,
    expansion_ggm,
    expansion_topo1,
    expansion_topo2,
):
    """
    Reproduce la parte equivalente a 'Restas_principal' del flujo antiguo,
    pero usando las expansiones ya calculadas previamente.
    """
    archivo_puntos = str(archivo_puntos)
    expansion_ggm = asegurar_txt(expansion_ggm)
    expansion_topo1 = asegurar_txt(expansion_topo1)
    expansion_topo2 = asegurar_txt(expansion_topo2)

    if not Path(archivo_puntos).exists():
        raise FileNotFoundError(f"No existe el archivo principal de puntos: {archivo_puntos}")
    if not Path(expansion_ggm).exists():
        raise FileNotFoundError(f"No existe la expansión GGM: {expansion_ggm}")
    if not Path(expansion_topo1).exists():
        raise FileNotFoundError(f"No existe la expansión TOPO1: {expansion_topo1}")
    if not Path(expansion_topo2).exists():
        raise FileNotFoundError(f"No existe la expansión TOPO2: {expansion_topo2}")

    df_restaurado = Restas_principal(
        Ruta_Archivo_Principal=archivo_puntos,
        Ruta_XGM2019_0_719=expansion_ggm,
        Ruta_EARTH_0_2159=expansion_topo2,
        Ruta_EARTH_0_719=expansion_topo1,
    )

    return df_restaurado


# ---------------------------------------------------------------------
# Restore iterativo
# ---------------------------------------------------------------------
def restore_iterativo(df: pd.DataFrame, tolerancia: float, max_iter: int) -> pd.DataFrame:
    """
    Calcula iterativamente la anomalía restaurada.
    """
    requeridas = ["Latitud", "altura", "Potencial_Perturbador", "anom_XGM", "anom_EARTH2014", "anom_ERTM"]
    faltantes = [c for c in requeridas if c not in df.columns]
    if faltantes:
        raise ValueError(f"Faltan columnas requeridas en el dataframe: {faltantes}")

    df = df.copy()

    df["gamma_0"] = gravedad_teorica_somigliana(df["Latitud"].values)
    df["gamma_h"] = gamma_h(df["Latitud"].values, df["gamma_0"].values, df["altura"].values)

    df["zeta"] = (
        (df["Potencial_Perturbador"] + df["anom_XGM"] + df["anom_EARTH2014"])
        / (df["gamma_h"] / 1e5)
        + df["anom_ERTM"]
    )

    for _ in range(int(max_iter)):
        zeta_old = df["zeta"].copy()
        Hn = df["altura"] - df["zeta"]

        df["gamma_h"] = gamma_h(df["Latitud"].values, df["gamma_0"].values, Hn.values)
        df["zeta"] = (
            (df["Potencial_Perturbador"] + df["anom_XGM"] + df["anom_EARTH2014"])
            / (df["gamma_h"] / 1e5)
            + df["anom_ERTM"]
        )

        if np.max(np.abs(df["zeta"] - zeta_old)) < tolerancia:
            break

    df["anom_res"] = df["zeta"]
    return df


# ---------------------------------------------------------------------
# Proceso completo
# ---------------------------------------------------------------------
def generacion_restore(
    archivo_puntos,
    expansion_ggm,
    expansion_topo1,
    expansion_topo2,
    grid_ertm,
    salida,
    tolerancia,
    max_iter,
):
    print("[INFO] Cargando datos y construyendo Restas_Principal")
    df = cargar_datos(
        archivo_puntos=archivo_puntos,
        expansion_ggm=expansion_ggm,
        expansion_topo1=expansion_topo1,
        expansion_topo2=expansion_topo2,
    )

    print("[INFO] Interpolando ERTM")
    df = interpolado_modelos(
        df=df,
        grid_path=grid_ertm,
        nombre_columna="anom_ERTM",
    )

    print("[INFO] Calculando restore iterativo")
    df = restore_iterativo(
        df=df,
        tolerancia=float(tolerancia),
        max_iter=int(max_iter),
    )

    salida = Path(salida)
    salida.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(
        salida,
        sep="\t",
        index=False,
        float_format="%.20f"
    )

    print(f"[OK] Restore guardado en: {salida}")


# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Restore sin ejecución de GravLab")

    parser.add_argument("archivo_puntos", help="Archivo principal de puntos")
    parser.add_argument("expansion_ggm", help="Expansión GGM")
    parser.add_argument("expansion_topo1", help="Expansión topográfica 0_719")
    parser.add_argument("expansion_topo2", help="Expansión topográfica 0_2159")
    parser.add_argument("grid_ertm", help="Grid ERTM")
    parser.add_argument("salida_restore", help="Archivo de salida Restore")
    parser.add_argument("tolerancia", type=float, help="Tolerancia")
    parser.add_argument("max_iter", type=int, help="Iteraciones máximas")

    args = parser.parse_args()

    generacion_restore(
        archivo_puntos=args.archivo_puntos,
        expansion_ggm=args.expansion_ggm,
        expansion_topo1=args.expansion_topo1,
        expansion_topo2=args.expansion_topo2,
        grid_ertm=args.grid_ertm,
        salida=args.salida_restore,
        tolerancia=args.tolerancia,
        max_iter=args.max_iter,
    )