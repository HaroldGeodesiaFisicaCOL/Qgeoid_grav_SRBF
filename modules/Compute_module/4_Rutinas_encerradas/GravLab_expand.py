#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
GravLab_expand.py

Programa independiente para generar las expansiones armónicas
usando Graflab.

Este script ejecuta exactamente la misma etapa que antes estaba
dentro de Restore1.py, pero ahora se ejecuta una sola vez en
el pipeline robusto.

Salida:
- Expansion_GGM
- Expansion_TOPO1
- Expansion_TOPO2
"""

import argparse
from pathlib import Path

# Funciones del proyecto
from Funciones_Auxiliares.Alistastamiento_Graflab import alistamiento_principal
from Funciones_Auxiliares.Proceso_remover_puntual import remover_puntual


def ejecutar_gravlab(
    carpeta_trabajo,
    graflab_path,
    ruta_puntos,
    ruta_datos_matlab,
    modelo_GGM,
    Expansion_GGM,
    modelo_TOPO,
    Expansion_TOPO1,
    Expansion_TOPO2
):

    carpeta_trabajo = Path(carpeta_trabajo)
    carpeta_trabajo.mkdir(parents=True, exist_ok=True)

    print("\n[INFO] Iniciando preparación para GravLab")

    # -----------------------------------------------------
    # 1) Preparación de archivos para GravLab
    # -----------------------------------------------------

    alistamiento_principal(
        str(ruta_puntos),
        str(ruta_datos_matlab)
    )

    print("[OK] Archivos preparados para GravLab")

    # -----------------------------------------------------
    # 2) Expansión armónica
    # -----------------------------------------------------

    print("[INFO] Ejecutando expansión armónica con GravLab")

    remover_puntual(
        path_gfc_model1=str(modelo_GGM),
        output_model1=str(Expansion_GGM),

        path_gfc_model2=str(modelo_TOPO),
        output_model2=str(Expansion_TOPO1),

        path_gfc_model3=str(modelo_TOPO),
        output_model3=str(Expansion_TOPO2),

        input_points=str(ruta_datos_matlab),

        graflab_path=str(graflab_path),

        out_dir=str(carpeta_trabajo)
    )

    print("[OK] Expansiones armónicas generadas")

    print("\nArchivos generados:")

    print(f"  - {Expansion_GGM}")
    print(f"  - {Expansion_TOPO1}")
    print(f"  - {Expansion_TOPO2}")


# ------------------------------------------------------------------
# MAIN
# ------------------------------------------------------------------

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Generación de expansiones armónicas con GravLab"
    )

    parser.add_argument("carpeta_trabajo", help="Directorio donde se ejecutará GravLab")
    parser.add_argument("graflab_path", help="Ruta donde está instalado GravLab")

    parser.add_argument("ruta_puntos", help="TXT de puntos centrales")
    parser.add_argument("ruta_datos_matlab", help="TXT de puntos para Matlab")

    parser.add_argument("modelo_GGM", help="Modelo GGM")
    parser.add_argument("Expansion_GGM", help="Archivo de salida expansión GGM")

    parser.add_argument("modelo_TOPO", help="Modelo topográfico EARTH2014")

    parser.add_argument("Expansion_TOPO1", help="Expansión topo grado bajo")
    parser.add_argument("Expansion_TOPO2", help="Expansión topo grado alto")

    args = parser.parse_args()

    ejecutar_gravlab(
        carpeta_trabajo=args.carpeta_trabajo,
        graflab_path=args.graflab_path,
        ruta_puntos=args.ruta_puntos,
        ruta_datos_matlab=args.ruta_datos_matlab,
        modelo_GGM=args.modelo_GGM,
        Expansion_GGM=args.Expansion_GGM,
        modelo_TOPO=args.modelo_TOPO,
        Expansion_TOPO1=args.Expansion_TOPO1,
        Expansion_TOPO2=args.Expansion_TOPO2
    )