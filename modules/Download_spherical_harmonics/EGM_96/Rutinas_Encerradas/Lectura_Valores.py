import pandas as pd

def lectura_valores(ruta):
    df = pd.read_csv(ruta,sep='\s+', header=None,names =['latitud','longitud','ondulacion'])
    df['longitud'] = df['longitud'] 
    return df