# =============================================================================
# Airborne gravity processing. This step include:
# - Spherical gaussian filter for the gravity observations by flight lines.
# - Subsampling of the gravity observations by flight lines. (Optional)
# - Spherical harmonics expansion of the background models. (GGM and Gtopo model)
# - Remove the long wavelength component and the short wavelength component of the gravity observations.
# - SRBF Bias correction of the gravity observations. (optional)
# - Outlier detection whit NMAD criteria.
# - Plotting the maps of the processed data.
# - Plotting the histograms of the processed data.
# =============================================================================
import os
import pandas as pd
from pathlib import Path

"""Funcion para realizar el filtrado Gaussiano esférico de aerogravimetría por líneas."""
from .Funciones_Auxiliares.FiltradoOptimoParser import GaussianFilter
""""Funcion para relaizar las diferencias del SaTop criollo con las perturbaciones Aéreas"""
from .Funciones_Auxiliares.Procesamiento_Aerogravimetricos import SaTop_Diferencias
""""Funcion para realizar el resampling de los datos o el submuestreo"""
from .Funciones_Auxiliares.Muestreador import sample_by_distance
"""
IMPORTE DE FUNCIONES PARA EL PROCESO DE REMOVER
"""
#se importa el bloque  de código el cuál genera el txt que se envía a graflab
from .Funciones_Auxiliares.Alistastamiento_Graflab import alistamiento_principal
#Se importa la la función que genera las órdeneas a Matlab para realizar al cálculo con los armónicos esféricos
from .Funciones_Auxiliares.Proceso_remover_puntual import remover_puntual
from .Funciones_Auxiliares.Restas_Remove import Restas_principal
#Funcion para generar las estadisticas de las perturbaciones.
from .Funciones_Auxiliares.Restas_Remove import estadisticos
#Funcion para realizar los mapas de los puntos.
from .Funciones_Auxiliares.Mapas_GMT import Mapa,Mapa_valor_gravedad

"""
IMPORTE DE LA FUNCION PARA EL CÁLCULO DE LOS SESGOS MEDIANTE SRBF
"""


# =============================================================================
# FUNCION NECESARIA PARA EL PASO 7.
# =============================================================================
from Funciones_Auxiliares.Paso7 import detect_outliers


import pandas as pd

def procesamiento_aereos(
    input_file,
    col_lat,
    col_lon,
    col_h,
    col_valor,
    path_gfc_model1,
    path_gfc_model2,
    path_gfc_model3,
    ruta_obs_final,
    graflab_path,
    filter_data: bool = False,
    filter_kwargs: dict | None = None,
    subsampling: bool = False,
    subsampling_kwargs: dict | None = None,

):
    base_dir = Path(__file__).resolve().parent

    remover_dir = base_dir / "Remover"
    visualizacion_dir = base_dir / "Visualizacion"
    archivo_final_dir = base_dir / "Archivo_Final"

    os.makedirs(remover_dir, exist_ok=True)
    os.makedirs(visualizacion_dir, exist_ok=True)
    os.makedirs(archivo_final_dir, exist_ok=True)

    ruta_datos_iniciales = Path(input_file).resolve()
    path_gfc_model1 = Path(path_gfc_model1).resolve()
    path_gfc_model2 = Path(path_gfc_model2).resolve()
    path_gfc_model3 = Path(path_gfc_model3).resolve()
    graflab_path = Path(graflab_path).resolve()

    datos_matlab = remover_dir / "Datos_Aereos_Matlab.txt"

    filter_kwargs = filter_kwargs or {}
    subsampling_kwargs = subsampling_kwargs or {}

    Aero_df = None

    # =========================================================================
    # Filtrado gaussiano de paso bajo
    # =========================================================================
    if filter_data:
        print("Gaussian filter low pass for the aerogravimetric data...")

        default_filter_args = {
            'input_file': ruta_datos_iniciales,
            'output_dir': 'Datos_filtrados/Datos_Aereos_Filtrado.txt',
            'project_column': "Proyecto",
            'line_column': "num_linea",
            'lat_column': "latitud",
            'lon_column': "longitud",
            'input_column': "Pre_Pert_Grav",
            'output_column': "Pre_Pert_Grav_Filt",
            'hwhm_km': 3.0,
            'earth_radius_km': 6378.137,
            'use_gpu': True,
            'gpu_threshold': 8000,
            'cpu_parallel_threshold': 800,
        }
        default_filter_args.update(filter_kwargs)
        Aero_df, reporte_filtro = GaussianFilter(**default_filter_args)
        print("Filtering done.")

    # =========================================================================
    # Submuestreo por distancia
    # =========================================================================
    if subsampling:
        if Aero_df is None:
            # No se filtró antes: cargamos el archivo base para poder
            # subsamplear de todos modos.
            print(f"No se aplicó filtro. Cargando '{input_file}' para submuestreo...")
            Aero_df = pd.read_csv(input_file, sep=None, engine='python')

        print("Subsampling the aerogravimetric data...")

        default_subsampling_args = {
            'df': Aero_df,
            'output_file': 'Submuestreo/Aero_grav_obs_Sin_NEXEN_2km.txt',
            'report_file': 'Submuestreo/Informe_Muestreo.txt',
        }
        default_subsampling_args.update(subsampling_kwargs)

        if 'sample_distance' not in default_subsampling_args:
            raise ValueError(
                "Debes indicar 'sample_distance' en subsampling_kwargs."
            )

        sample_by_distance(**default_subsampling_args)
        print("Subsampling done.")

    alistamiento_principal(input_file,col_lat=col_lat,col_long=col_lon,col_altura=col_h,
                        ruta_sal=str(datos_matlab))
    remover_puntual(out_dir=str(remover_dir),
                    path_gfc_model1=str(path_gfc_model1),
                    output_model1=str(remover_dir / "XGM2019_0_719"),
                    path_gfc_model2=str(path_gfc_model2),
                    output_model2=str(remover_dir / "dv_ell_earth2014_0_719"),
                    path_gfc_model3=str(path_gfc_model3),
                    output_model3=str(remover_dir / "dv_ell_earth2014_0_2159"),
                    input_points=str(datos_matlab),
                    graflab_path=str(graflab_path))

    # Ahora se realiza el proceso de las restas ahora que se tienen las componentes de
    # Longitud de onda larga y longitud de onda corta
    df_removido = Restas_principal(
        Ruta_Archivo_Principal=str(ruta_datos_iniciales),
        col_valor=col_valor,
        col_altura=col_h,
        Ruta_XGM2019_0_719=str(remover_dir / "XGM2019_0_719.txt"),
        Ruta_EARTH_0_2159=str(remover_dir / "dv_ell_earth2014_0_2159.txt"),
        Ruta_EARTH_0_719=str(remover_dir / "dv_ell_earth2014_0_719.txt"),)

    df_removido, df_outliers, epsilon_global = detect_outliers(
        df_removido,
        lat_col=col_lat,
        lon_col=col_lon,
        val_col=col_valor,
    )
    print(f"Global epsilon: {epsilon_global}")

    # Ploteo de las estadísticas del proceso de remover.
    arreglo =[ col_valor,'Pert_GGM','Perturbaciones_residuales']
    labels =['$\\delta_g$','$\\delta_g - \\delta_gGGM$','$\\delta_g - \\delta_gGGM - \\delta_gGtopo$']
    estadisticos(df=df_removido, columnas=arreglo, labels=labels)

    df_removido.to_csv(archivo_final_dir / "Archivo_Final/Datos_Aereos.txt", sep="\t", index=False)

    df_removido[[col_lat, col_lon, col_h, col_valor,"ERTM2160","XGM2019","EARTH2014","Pert_GGM",'Perturbaciones_residuales']].to_csv(
        ruta_obs_final,
        sep="\t",
        index=False,
    )

    region = [
        df_removido[col_lon].min()-0.3,
        df_removido[col_lon].max()+0.3,
        df_removido[col_lat].min()-0.3,
        df_removido[col_lat].max()+0.3,
    ]

    # =============================================================================
    # Maps GMT of remove procedure of the Airborne data
    # =============================================================================

    df_limpio_clipp = df_removido
    df_limpio_clipp[col_valor] = df_limpio_clipp[col_valor].clip(lower=-100, upper=100)
    df_limpio_clipp["Pert_GGM"] = df_limpio_clipp["Pert_GGM"].clip(lower=-40, upper=40)
    df_limpio_clipp["Perturbaciones_residuales"] = df_limpio_clipp["Perturbaciones_residuales"].clip(lower=-40, upper=40)

    # Ploteo del comportamiento del proceso de remover.
    Mapa_valor_gravedad(
         region=region,
         df=df_limpio_clipp,
         col_lat=col_lat,
         col_lon=col_lon,
         zmin=-100,
         zmax=100,
         Ruta_Mapa=str(visualizacion_dir / "Perturbaciones_observadas_clip.png"),
         Columna=col_valor,
     )

    Mapa_valor_gravedad(
         region=region,
         df=df_limpio_clipp,
         col_lat=col_lat,
         col_lon=col_lon,
         zmin=-60,
         zmax=60,
         Ruta_Mapa=str(visualizacion_dir / "Pert_GGM_clip.png"),
         Columna="Pert_GGM",
     )

    Mapa_valor_gravedad(
         region=region,
         df=df_limpio_clipp,
         col_lat=col_lat,
         col_lon=col_lon,
         zmin=-60,
         zmax=60,
         Ruta_Mapa=str(visualizacion_dir / "Perturbaciones_residuales_clip.png"),
         Columna="Perturbaciones_residuales",
     )
