import pandas as pd
import os

def lectura_txt(ruta):
    return pd.read_csv(ruta,sep='\t',engine='python')

def Alistamiento(df,arreglo,ruta):
    df_listo = df[arreglo]
    # df_listo[arreglo[1]] = df_listo[arreglo[1]] +360
    os.makedirs('Remover',exist_ok=True)
    df_listo.to_csv(ruta,sep='\t', index=False, header=False)

def alistamiento_principal(ruta_datos,col_lat,col_long,col_altura,ruta_sal):
    df_terrestres = lectura_txt(ruta_datos)
    print(df_terrestres.columns)
    arreglo=[col_lat,col_long,col_altura]
    df_terrestres = df_terrestres[arreglo]
    Alistamiento(df_terrestres,arreglo,ruta_sal)
