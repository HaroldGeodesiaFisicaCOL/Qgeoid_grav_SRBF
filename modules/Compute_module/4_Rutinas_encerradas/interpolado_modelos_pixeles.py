"Da altura elipsoidal a los pixeles usando el SRTM y el EGM2008"

import pygmt
import geopandas as gpd
import pandas as pd 
import numpy as np
import rasterio
from rasterio.transform import from_origin
from rasterio.crs import CRS
 
## FUNCIONES NECESARIAS PARA EL SCRIPT 
def interpolado_MODELOS(df,srtm_path,EGM2008_path,nombre_columna_SRTM,nombre_columna_EGM2008):
    # Leer el archivo .txt con pandas
    coordenadas = list(zip(df['Longitud'], df['Latitud']))
    # Realizar la interpolación para el modelo SRTM
    interpolated_values_SRTM = pygmt.grdtrack(points=coordenadas, grid=srtm_path, interpolation='l')
    # Realizar la interpolación para el modelo EGM2008 30mx30m
    interpolated_values_EGM2008 = pygmt.grdtrack(points=coordenadas, grid=EGM2008_path, interpolation='l')
    
    # Asignar los valores interpolados a las columnas correspondientes en el DataFrame
    df[nombre_columna_SRTM] = interpolated_values_SRTM[2]
    df[nombre_columna_EGM2008] = interpolated_values_EGM2008[2]
    df['altura']=df[nombre_columna_SRTM]+df[nombre_columna_EGM2008]
    
def lectura_txt(ruta):
    df = pd.read_csv(ruta, sep=r"\s+")
    return df
 
### EJECUCION DEL SCRIPT 
def ejecucion_interpolado(ruta_pixeles,srtm_path,EGM2008_path):
    df = lectura_txt(ruta_pixeles)
    interpolado_MODELOS(df,srtm_path,EGM2008_path,nombre_columna_SRTM='SRTM',nombre_columna_EGM2008='EGM2008')
    df.to_csv(ruta_pixeles, sep='\t', index=False, header=True,float_format="%.20f")
 
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Ejecución del interpolado en los pixeles del modelo")
    parser.add_argument("ruta_pixeles", type=str, help="Ruta donde se encuentran los pixeles")
    parser.add_argument("srtm_path", type=str, help="Ruta del modelo SRTM30")
    parser.add_argument("EGM2008_path", type=str, help="Ruta del modelo SAM Qgeoide")
    args = parser.parse_args()
    ejecucion_interpolado(ruta_pixeles= args.ruta_pixeles, 
                          srtm_path = args.srtm_path, 
                          EGM2008_path = args.EGM2008_path)