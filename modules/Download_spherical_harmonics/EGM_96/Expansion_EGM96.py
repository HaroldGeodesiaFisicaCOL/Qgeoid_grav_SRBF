# =============================================================================
# IMPORTE DE LAS FUNCIONES NECESARIAS
# =============================================================================
from pathlib import Path

from .Rutinas_Encerradas.Generacion_Raster import generacion_raster
from .Rutinas_Encerradas.Lectura_Valores import lectura_valores
from .Rutinas_Encerradas.Proceso_remover_puntual import remover_puntual


def expansion_EGM96(
    lat_min,
    lat_max,
    long_min,
    long_max,
    out_dir,
    path_gfc_model1,
    output_model1,
    Nombre_mat,
    DTM_path,
    Raster_EGM96_path,
    graflab_path,
    espaciado=1 / 60,
):
    remover_puntual(
        out_dir=out_dir,
        path_gfc_model1=path_gfc_model1,
        output_model1=output_model1,
        Nombre_mat=Nombre_mat,
        lat_max=lat_max,
        lat_min=lat_min,
        long_max=long_max,
        long_min=long_min,
        espaciado=espaciado,
        DTM=DTM_path,
        graflab_path=graflab_path,
    )

    # Se realiza la lectura de los valores de anomalía de altura expandidos.
    root = Path(out_dir).expanduser().resolve().parent
    output_txt = Path(output_model1)

    if not output_txt.suffix:
        output_txt = output_txt.with_suffix(".txt")

    if not output_txt.is_absolute():
        output_txt = (root / output_txt).resolve()

    print(">> Leyendo expansión EGM96:", output_txt)

    expansion = lectura_valores(output_txt)

    # Ahora se genera el raster del modelo expandido por armónicos esféricos.
    generacion_raster(
        puntos=expansion,
        col_lat="latitud",
        col_long="longitud",
        tam_pixel=1,
        nombre_raster=Raster_EGM96_path,
        nombre_col="ondulacion",
    )
