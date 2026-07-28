import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
from Funciones_Auxiliares.Tratamiento_Alturas import interpolado_MODELOS


#FUNCIONES NECESARIAS PARA EL SCRIPT

def lectura_txt(ruta):
    df = pd.read_csv(ruta, sep=r'\s+', engine='python', header=None)
    df.columns = ['latitud', 'longitud', 'altura', 'perturbacion']
    return df

def lectura_txt2(ruta):
    return pd.read_csv(ruta,sep='\t',index_col=False)

def componente_onda_corta(df_719,df_5481):
    df_720_5480 = df_5481['perturbacion']-df_719['perturbacion']
    return df_720_5480

def remover(df_concatenado):
    df_concatenado['Pert_GGM'] = df_concatenado['perturbacion_gravedad']- df_concatenado['XGM2019']
    df_concatenado['Perturbaciones_residuales'] = df_concatenado['perturbacion_gravedad']- df_concatenado['XGM2019'] - df_concatenado['EARTH2014'] - df_concatenado['ERTM2160']
def correcion_atmosferica(df):
    correcion = 0.874 - 9.9e-5 * df['altura'] + 3.56e-9*df['altura']**2
    df['Perturbaciones_residuales'] =df['Perturbaciones_residuales']+correcion       
def estadisticos(df, columnas, labels):
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": "cm"
    })
    for col, label in zip(columnas, labels):
        datos = df[col].dropna()  # Por si hay NaN

        # Estadísticos
        media = np.mean(datos)
        std = np.std(datos)
        print(f'\nEstadísticas para la columna "{col}":')
        print(f'  Media   : {media:.4f}')
        print(f'  Std     : {std:.4f}')
        print(f'  Máximo  : {np.max(datos):.4f}')
        print(f'  Mínimo  : {np.min(datos):.4f}')
        print(f'  Mediana : {np.median(datos):.4f}')
        print('-----------------------------------')

        # Histograma
        plt.figure(figsize=(7, 4))
        count, bins, ignored = plt.hist(datos, bins=20, color='skyblue', edgecolor='black', density=True)

        # Curva normal superpuesta
        x = np.linspace(min(bins), max(bins), 1000)
        plt.plot(x, norm.pdf(x, media, std), 'r-', lw=2, label='Curva normal')

        plt.xlabel(label)
        plt.ylabel('Densidad')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()

        # Guardar el plot usando el nombre de la columna
        filename = f'Ploteo_{col}.pdf'
        #plt.savefig(filename)
        print(f'Guardado: {filename}')
        plt.show()

def Restas_principal(Ruta_Archivo_Principal,Ruta_XGM2019_0_719,Ruta_EARTH_0_2159,Ruta_EARTH_0_719,Ruta_ERTM2160):
    df_Original = lectura_txt2(Ruta_Archivo_Principal)
    df_XGM2019_0_719 = lectura_txt(Ruta_XGM2019_0_719)
    df_EARTH_0_2159 = lectura_txt(Ruta_EARTH_0_2159)
    df_EARTH_0_719 = lectura_txt(Ruta_EARTH_0_719)
    
    #comportamiento onda corta
    df_720_2159 = componente_onda_corta(df_719=df_EARTH_0_719, df_5481=df_EARTH_0_2159)
    df_Original = interpolado_MODELOS(df=df_Original, modelo_path=Ruta_ERTM2160, 
                        nombre_columna_modelo='ERTM2160')
    
    df_Original['XGM2019'] = df_XGM2019_0_719['perturbacion']
    df_Original['EARTH2014'] = df_720_2159
    remover(df_Original)
    correcion_atmosferica(df_Original)
    return df_Original
    

