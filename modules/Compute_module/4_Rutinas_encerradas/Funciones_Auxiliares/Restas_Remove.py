import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm


#FUNCIONES NECESARIAS PARA EL SCRIPT

def lectura_txt(ruta):
    df = pd.read_csv(ruta, sep=r'\s+', engine='python', header=None)
    df.columns = ['latitud', 'longitud', 'altura', 'valor']
    return df

def lectura_txt2(ruta):
    return pd.read_csv(ruta,sep='\t',index_col=False)

def componente_onda_corta(df_719,df_5481):
    df_720_5480 = df_5481['valor']-df_719['valor']
    return df_720_5480     

def Restas_principal(Ruta_Archivo_Principal,Ruta_XGM2019_0_719,Ruta_EARTH_0_2159,Ruta_EARTH_0_719):
    df_Original = lectura_txt2(Ruta_Archivo_Principal)
    df_XGM2019_0_719 = lectura_txt(Ruta_XGM2019_0_719)
    df_EARTH_0_2159 = lectura_txt(Ruta_EARTH_0_2159)
    df_EARTH_0_719 = lectura_txt(Ruta_EARTH_0_719)
    
    #comportamiento onda corta
    df_720_2159 = componente_onda_corta(df_719=df_EARTH_0_719, df_5481=df_EARTH_0_2159)
    
    df_Original['anom_XGM'] = df_XGM2019_0_719['valor']
    df_Original['anom_EARTH2014'] = df_720_2159
    return df_Original
    

