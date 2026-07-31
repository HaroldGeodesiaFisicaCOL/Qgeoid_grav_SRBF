# -*- coding: utf-8 -*-
"""
Created on Fri May  2 16:56:57 2025

@author: harold.olarte
"""

import pygmt
import geopandas as gpd
import pandas as pd 
import numpy as np
import rasterio
from rasterio.transform import from_origin
from rasterio.crs import CRS
from shapely.geometry import Point
import os
import matplotlib.pyplot as plt

Uo = 62636860.850 #POTENCIAL NORMAL GRS80
Wo = 62636853.4  #POTENCIAL IHRF
GM_GGM=3.986004415*10**14
GM_GRS80=3.986005*10**14
gamma_a =978032.67715
gamma_b=983218.63685
c1 = 0.0053024
c2=  0.0000058
f =  0.00335281068118
m = 0.00344978600308
a=6378137
b=6356752.3141 
e2=0.00669438002290 

plt.rcdefaults()
plt.rcParams['text.usetex'] = False  # Desactivar LaTeX
plt.rcParams["font.family"] = "serif"

def txt_a_geodataframe(df, epsg=4686, separador='\t'):
    geometry = [Point(xy) for xy in zip(df['Longitud'], df['Latitud'])]
    gdf = gpd.GeoDataFrame(df, geometry=geometry, crs=f"EPSG:{epsg}")
    return gdf       
def interpolado_MODELOS(df,goco_path,
                        srtm_gravity_path,
                        nombre_columna_GOCO,
                        nombre_columna_SRTMGrav):
    
    coordenadas = list(zip(df['Longitud'], df['Latitud']))
    interpolated_values_GOCO = pygmt.grdtrack(points=coordenadas, grid=goco_path, interpolation='l')
    interpolated_values_SRTMGrav = pygmt.grdtrack(points=coordenadas, grid=srtm_gravity_path, interpolation='l')
    # Asignar los valores interpolados a las columnas correspondientes en el DataFrame
    df[nombre_columna_GOCO] = interpolated_values_GOCO[2]
    df[nombre_columna_SRTMGrav] = interpolated_values_SRTMGrav[2]
    return df
    
def remove(df,columna,nombre_columna_GOCO,nombre_columna_SRTMGrav):
    df['Diferencias_SATOP']=df[columna]-df[nombre_columna_GOCO]-df[nombre_columna_SRTMGrav]
    return df

def media_por_proyecto(df):
    estadisticas_por_proyecto = df.groupby('Proyecto')['Diferencias_SATOP'].agg(['mean', 'std']).reset_index()
    return estadisticas_por_proyecto

def ploteo_grafica(estadisticas_por_proyecto,ruta_grafica):
    estadisticas_por_proyecto['Proyecto_Index'] = np.arange(len(estadisticas_por_proyecto))
    # Crear el gráfico
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    # Gráfico de barras para la media
    ax1.bar(estadisticas_por_proyecto['Proyecto'], estadisticas_por_proyecto['mean'], color='darkcyan', label='Media', alpha=1)
    
    # Configurar el primer eje (barras)
    ax1.set_xlabel('Proyecto Aerogravimétrico')
    ax1.set_ylabel('Media de Diferencias mGals', color='darkcyan')
    ax1.tick_params(axis='y', labelcolor='darkcyan')
    
    # Crear el segundo eje para la desviación estándar
    ax2 = ax1.twinx()
    ax2.plot(estadisticas_por_proyecto['Proyecto'], estadisticas_por_proyecto['std'], color='black', marker='o', label='Desviación Estándar')
    ax2.set_ylabel('Std de Diferencias mGls', color='black')
    ax2.tick_params(axis='y', labelcolor='black')
    
    # Añadir título y mostrar el gráfico
    #plt.title('Media y Desviación Estándar de Diferencias SATOP por Proyecto Aerogravimétrico')
    plt.xticks(estadisticas_por_proyecto['Proyecto_Index'], [str(i+1) for i in estadisticas_por_proyecto['Proyecto_Index']])
    plt.tight_layout()
    plt.grid()
    plt.savefig(ruta_grafica)
    # Mostrar el gráfico
    plt.show()

# Ejecución del Script

def SaTop_Diferencias(df,columna,goco_path,srtm_gravity_path,resultado_diferencias_path,ruta_grafica):
    df = txt_a_geodataframe(df)
    df=interpolado_MODELOS(df,  
                        goco_path=goco_path,
                        srtm_gravity_path=srtm_gravity_path,
                        nombre_columna_GOCO='Pert_Goco', 
                        nombre_columna_SRTMGrav='Pert_SRTMGrav')
    
    df=remove(df,columna,nombre_columna_GOCO='Pert_Goco', 
           nombre_columna_SRTMGrav='Pert_SRTMGrav')
    
    valores_medios_diferencias=media_por_proyecto(df)
    ploteo_grafica(valores_medios_diferencias,ruta_grafica)
    valores_medios_diferencias.to_csv(resultado_diferencias_path,sep='\t')
    proyecto = valores_medios_diferencias.loc[valores_medios_diferencias["mean"].abs().idxmin(), "Proyecto"]
    return df,proyecto



