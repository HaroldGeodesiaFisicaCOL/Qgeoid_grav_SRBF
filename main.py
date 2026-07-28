# In this script, the first part consist in downloading the spherical harmonics coefficients of the
# geopotential models, long wavelegth (GGM) and short wavelegth (Gtopo).
import os
import numpy as np
from pathlib import Path


from modules.Download_spherical_harmonics.Dowloand_grid_Model import (
    dowloand_all_folder,
    download_tiles,
    generate_download_list,
    generate_download_ERTM2160
)
from modules.Download_spherical_harmonics.ERTM2160 import (
    generar_modelo_ertm2160
)
from modules.Download_spherical_harmonics.Download_spherical_harmonics import (
    descargar_modelo_icgem,
)

from modules.Download_spherical_harmonics.Dowloand_SRTM import (
    generar_modelo_srtm,
)

from modules.Download_spherical_harmonics.EGM_96.Expansion_EGM96 import (
    expansion_EGM96,
)

from modules.Remove_module.Terrestrial_processing.Main_Procesamiento_terrestres import (
    procesamiento_terrestres,
)
os.makedirs('modules/Compute_module/1_Modelos/modeloXGM2019',exist_ok=True)
os.makedirs('modules/Compute_module/1_Modelos/modelo_dv_ell_Earth2014',exist_ok=True)
GGM = 'XGM2019'
Gtopo1 = 'dV_ELL_Earth2014_plusGRS80'
Gtopo2 = 'dV_ELL_Earth2014_5480_plusGRS80'
GGM_EGM96 = 'EGM96'
print('Dowloand the GGM model...')
descargar_modelo_icgem(modelo=GGM,grado=760,
                            ruta='modules/Compute_module/1_Modelos/modeloXGM2019',
                            tipo="global",
                            sobrescribir=False,
                            timeout=120,
                            max_reintentos=8,)
print('Dowloand the Gtopo model dv_ELL_Earth2014 d/o 2190...')
descargar_modelo_icgem(modelo=Gtopo1,grado=2190,
                            ruta='modules/Compute_module/1_Modelos/modelo_dv_ell_Earth2014',
                            tipo="topografia",
                            sobrescribir=False,
                            timeout=120,
                            max_reintentos=8,)
print('Dowloand the Gtopo model dv_ELL_Earth2014 d/o 5480...')
descargar_modelo_icgem(modelo=Gtopo2,grado=5480,
                            ruta='modules/Compute_module/1_Modelos/modelo_dv_ell_Earth2014',
                            tipo="topografia",
                            sobrescribir=False,
                            timeout=120,
                            max_reintentos=8,)
print('Dowloand the Gtopo model ERTM2190...')
#This parameters define the area to download
lat_min = 34
lat_max = 50
lon_min = 5
lon_max = 25
# Dowloand the model ERTM2160's zeta component
# base_url_zeta = "https://ddfe.curtin.edu.au/gravitymodels/ERTM2160/data/geoid"
base_url_zeta = "https://ddfe.blazejbucha.com/models/ERTM2160/data/geoid/"
base_url_gravity = "https://ddfe.blazejbucha.com/models/ERTM2160/data/dg/"
base_local_path_zeta = r"modules/Compute_module/1_Modelos/modeloERTM2160/data/geoid"
base_local_path_gravity = r"modules/Compute_module/1_Modelos/modeloERTM2160/data/dg"
os.makedirs(base_local_path_zeta, exist_ok=True)
os.makedirs(base_local_path_gravity, exist_ok=True)

print('Downloading ERTM2160 geoid component...')
# generate_download_ERTM2160(lat_min, lat_max, lon_min, lon_max, base_url_zeta, base_local_path_zeta)
print('Downloading ERTM2160 gravity component...')
# generate_download_ERTM2160(lat_min, lat_max, lon_min, lon_max, base_url_gravity, base_local_path_gravity)

# generate the grid model of height anomaly (m)
generar_modelo_ertm2160(
    'geoid', lon_min,lon_max, lat_min, lat_max,
    1.0, 1.0, "modules/Compute_module/1_Modelos/modeloERTM2160/data", np.nan, 'cubic',
    outputfile='modules/Compute_module/1_Modelos/modeloERTM2160/Anomalias_Altura_ERTM_2160.tif'
    )

generar_modelo_ertm2160(
    'gravity', lon_min,lon_max, lat_min, lat_max,
    1.0, 1.0, "modules/Compute_module/1_Modelos/modeloERTM2160/data", np.nan, 'cubic',
    outputfile='modules/Compute_module/1_Modelos/modeloERTM2160/Perturbaciones_Gravedad_ERTM_2160.tif'
    )


# DOWLOAND THE SRTM MODEL
print('Downloading SRTM model...')
os.makedirs("modules/Compute_module/1_Modelos/modelo_SRTM/data", exist_ok=True)
# generar_modelo_srtm(
#     lon_min, lon_max, lat_min, lat_max,
# )
# Delete the temporary folder to models ERTM and SRTM
# os.rmdir("modules/Compute_module/1_Modelos/modelo_SRTM/data")
# os.rmdir("modules/Compute_module/1_Modelos/modeloERTM2160/data")

# Spherical harmonics synthesis modelo EGM96
print('Dowloand the GGM EGM96 model for the heights whit SRTM model...')
descargar_modelo_icgem(modelo=GGM_EGM96,grado=360,
                            ruta='modules/Compute_module/1_Modelos/modelo_EGM2008',
                            tipo="global",
                            sobrescribir=False,
                            timeout=120,
                            max_reintentos=8,)
# Spherical harmonics expansion modelo EGM96
print('Expanding the GGM EGM96 model for the heights whit SRTM model...')
project_root = Path(__file__).resolve().parent
graflab_path = project_root / "dependencies" / "Graflab"

expansion_EGM96(
    lat_min=lat_min,
    lat_max=lat_max,
    long_min=lon_min,
    long_max=lon_max,
    path_gfc_model1="modelo_EGM2008/EGM96.gfc",
    output_model1="modelo_EGM2008/EGM96",
    out_dir="modules/Compute_module/1_Modelos/modelo_EGM2008",
    Nombre_mat="EGM96.m",
    DTM_path=str(project_root / "dependencies" / "Graflab" / "Earth2014_SUR2014_10800.mat"),
    Raster_EGM96_path="modules/Compute_module/1_Modelos/modelo_EGM2008/EGM96.tif",
    graflab_path=str(graflab_path),
    espaciado=1 / 60,
)


# REMOVE STEP FOR THE GRAVITY OBSERVATIONS
# First, the user have to know the number of technique's  gravity observations
# For example, in the case of Italian model we have 2 techniques: Terrestrial and aereal.

# For the terrestrial technique :
# Include the column names for latitude, longitude, height, and value
print('Removing terrestrial data...')
T_col_lat = "Lat"
T_col_lon = "Lon"
T_col_h = "h"
T_col_valor = r"gravity disturbance(mGal)"
procesamiento_terrestres(
    ruta_datos_iniciales="modules/Remove_module/Initial_observation/terrestrial_data.txt",
    col_lat=T_col_lat,
    col_lon=T_col_lon,
    col_h=T_col_h,
    col_valor=T_col_valor,
    path_gfc_model1="modules/Compute_module/1_Modelos/modeloXGM2019/XGM2019.gfc",
    path_gfc_model2="modules/Compute_module/1_Modelos/modelo_dv_ell_Earth2014/dV_ELL_Earth2014_plusGRS80.gfc",
    path_gfc_model3="modules/Compute_module/1_Modelos/modelo_dv_ell_Earth2014/dV_ELL_Earth2014_plusGRS80.gfc",
    Ruta_ERTM2160="modules/Compute_module/1_Modelos/modeloERTM2160/Perturbaciones_Gravedad_ERTM_2160.tif",
    ruta_obs_final="modules/Compute_module/3_Observaciones/terrestrial_data.txt",
    graflab_path=str(graflab_path),
)
