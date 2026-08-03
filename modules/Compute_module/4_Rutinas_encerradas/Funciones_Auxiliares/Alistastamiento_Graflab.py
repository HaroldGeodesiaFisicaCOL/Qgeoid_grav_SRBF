import pandas as pd
import os
import numpy as np

def lectura_txt(ruta):
    return pd.read_csv(ruta,sep='\t',engine='python')

def Alistamiento(df,ruta_datos_matlab):
    arreglo=['Latitud','Longitud','altura']
    df_listo = df[arreglo]
    # df_listo['Longitud'] = df_listo['Longitud'] +360
    # df_listo['altura'] = np.zeros(len(df_listo))
    os.makedirs('Remover',exist_ok=True)
    df_listo.to_csv(ruta_datos_matlab,sep='\t', index=False, header=False)

def alistamiento_principal(ruta_datos,ruta_datos_matlab):
    df_terrestres = lectura_txt(ruta_datos)
    arreglo=['Latitud','Longitud','altura']
    df_terrestres = df_terrestres[arreglo]
    Alistamiento(df_terrestres,ruta_datos_matlab)
