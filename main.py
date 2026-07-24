# In this script, the first part consist in downloading the spherical harmonics coefficients of the
# geopotential models, long wavelegth (GGM) and short wavelegth (Gtopo).
import os

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
