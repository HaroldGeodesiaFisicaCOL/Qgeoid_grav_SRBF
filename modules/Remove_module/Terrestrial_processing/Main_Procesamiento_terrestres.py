# =============================================================================
# Terrestrial gravity processing. This step include:
# - Spherical harmonics expansion of the background models. (GGM and Gtopo model)
# - Remove the long wavelength component and the short wavelength component of the gravity observations.
# - Outlier detection whit NMAD criteria.
# - Plotting the maps of the processed data.
# - Plotting the histograms of the processed data.
# =============================================================================
import os
from pathlib import Path

from pandas.core.col import col

"""
IMPORTE DE FUNCIONES PARA EL PROCESO DE REMOVER
"""
# Se importa la funcion que calcula las perturbaciones de gravedad
from .Funciones_Auxiliares.gamma_teorico import calculo_perturbaciones
# se importa el bloque de código el cuál genera el txt que se envía a graflab
from .Funciones_Auxiliares.Alistastamiento_Graflab import alistamiento_principal
# Se importa la función que genera las órdenes a Matlab para realizar el cálculo con los armónicos esféricos
from .Funciones_Auxiliares.Proceso_remover_puntual import remover_puntual
# Funcion para realizar los mapas de los puntos.
from .Funciones_Auxiliares.Mapas_GMT import Mapa_valor_gravedad
# Funcion para realizar las restas del remove.
from .Funciones_Auxiliares.Restas_Remove import Restas_principal
# Funcion para generar las estadisticas de las perturbaciones.
from .Funciones_Auxiliares.Restas_Remove import estadisticos
from .Funciones_Auxiliares.Paso7 import detect_outliers


# =============================================================================
# Script execution
# =============================================================================
def procesamiento_terrestres(
    ruta_datos_iniciales,
    col_lat,
    col_lon,
    col_h,
    col_valor,
    path_gfc_model1,
    path_gfc_model2,
    path_gfc_model3,
    Ruta_ERTM2160,
    ruta_obs_final,
    graflab_path,
):
    base_dir = Path(__file__).resolve().parent

    remover_dir = base_dir / "Remover"
    visualizacion_dir = base_dir / "Visualizacion"
    archivo_final_dir = base_dir / "Archivo_Final"

    os.makedirs(remover_dir, exist_ok=True)
    os.makedirs(visualizacion_dir, exist_ok=True)
    os.makedirs(archivo_final_dir, exist_ok=True)

    ruta_datos_iniciales = Path(ruta_datos_iniciales).resolve()
    path_gfc_model1 = Path(path_gfc_model1).resolve()
    path_gfc_model2 = Path(path_gfc_model2).resolve()
    path_gfc_model3 = Path(path_gfc_model3).resolve()
    Ruta_ERTM2160 = Path(Ruta_ERTM2160).resolve()
    graflab_path = Path(graflab_path).resolve()

    datos_matlab = remover_dir / "Datos_Terrestres_Matlab.txt"
    calculo_perturbaciones(
        str(ruta_datos_iniciales),
        col_grav=col_valor,
        col_lat=col_lat,
        col_h=col_h,
    )
    alistamiento_principal(
        str(ruta_datos_iniciales),
        col_lat,
        col_lon,
        col_h,
        ruta_sal=str(datos_matlab),
    )

    remover_puntual(
        out_dir=str(remover_dir),
        path_gfc_model1=str(path_gfc_model1),
        output_model1=str(remover_dir / "XGM2019_0_719"),
        path_gfc_model2=str(path_gfc_model2),
        output_model2=str(remover_dir / "dv_ell_earth2014_0_719"),
        path_gfc_model3=str(path_gfc_model3),
        output_model3=str(remover_dir / "dv_ell_earth2014_0_2159"),
        input_points=str(datos_matlab),
        graflab_path=str(graflab_path),
    )

    # Ahora se realiza el proceso de las restas ahora que se tienen las componentes de
    # Longitud de onda larga y longitud de onda corta
    df_removido = Restas_principal(
        Ruta_Archivo_Principal=str(ruta_datos_iniciales),
        col_lat=col_lat,
        col_lon=col_lon,
        col_valor=col_valor,
        col_altura=col_h,
        Ruta_XGM2019_0_719=str(remover_dir / "XGM2019_0_719.txt"),
        Ruta_EARTH_0_2159=str(remover_dir / "dv_ell_earth2014_0_2159.txt"),
        Ruta_EARTH_0_719=str(remover_dir / "dv_ell_earth2014_0_719.txt"),
        Ruta_ERTM2160=str(Ruta_ERTM2160),
    )

    df_removido, df_outliers, epsilon_global = detect_outliers(
        df_removido,
        lat_col=col_lat,
        lon_col=col_lon,
        val_col=col_valor,
    )

    print(f"Global epsilon: {epsilon_global}")

    # Ploteo de las estadísticas del proceso de remover.
    arreglo = [
        col_valor,
        "Pert_GGM",
        "Perturbaciones_residuales",
    ]

    labels = [
        "$\\delta_g$",
        "$\\delta_g - \\delta_gGGM$",
        "$\\delta_g - \\delta_gGGM - \\delta_gGtopo$",
    ]

    estadisticos(
        df=df_removido,
        columnas=arreglo,
        labels=labels,
    )

    df_removido.to_csv(
        archivo_final_dir / "Datos_Terrestres.txt",
        sep="\t",
        index=False,
    )
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
    # Maps GMT of remove procedure of the terrestrial data
    # =============================================================================

    df_limpio_clipp = df_removido

    df_limpio_clipp[col_valor] = df_limpio_clipp[col_valor].clip(
        lower=-100,
        upper=100,
    )
    df_limpio_clipp["Pert_GGM"] = df_limpio_clipp["Pert_GGM"].clip(
        lower=-60,
        upper=60,
    )
    df_limpio_clipp["Perturbaciones_residuales"] = df_limpio_clipp[
        "Perturbaciones_residuales"
    ].clip(
        lower=-60,
        upper=60,
    )

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
