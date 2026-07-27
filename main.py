# In this script, the first part consist in downloading the spherical harmonics coefficients of the
# geopotential models, long wavelegth (GGM) and short wavelegth (Gtopo).
import os
import numpy as np
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

os.makedirs('modules/Compute_module/1_Modelos/modeloXGM2019',exist_ok=True)
os.makedirs('modules/Compute_module/1_Modelos/modelo_dv_ell_Earth2014',exist_ok=True)
GGM = 'XGM2019'
Gtopo1 = 'dV_ELL_Earth2014_plusGRS80'
Gtopo2 = 'dV_ELL_Earth2014_5480_plusGRS80'
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
base_local_path_zeta = r"modules/Compute_module/1_Modelos/modeloERTM2160/data/geoid"
os.makedirs(base_local_path_zeta, exist_ok=True)
generate_download_ERTM2160(lat_min, lat_max, lon_min, lon_max, base_url_zeta, base_local_path_zeta)
# generate the grid model of height anomaly (m)
X1, Y1, Z1 = generar_modelo_ertm2160(
    'geoid', lon_min,lon_max, lat_min, lat_max,
    1.0, 1.0, "modules/Compute_module/1_Modelos/modeloERTM2160/data", np.nan, 'cubic',
    outputfile='modules/Compute_module/1_Modelos/modeloERTM2160/Anomalias_Altura_ERTM_2160.tif'
    )
