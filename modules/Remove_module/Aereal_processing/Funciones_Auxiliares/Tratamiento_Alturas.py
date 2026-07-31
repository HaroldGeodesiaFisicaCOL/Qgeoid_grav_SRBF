import pygmt
import numpy as np
import pandas as pd
# CONSTANTES USADAS
gamma_a=978032.67715
gamma_b=983218.63685
c1=0.0053024
c2=0.0000058
a=6378137
b=6356752.3141
E=521854.0097
w2=5.317494117*(10**-9)
KM=3.986005*(10**14)
m=0.00344978600308
f=0.00335281068118

def datos_Null(df,columna):
    df_datos_fuera_modelo = df[df[columna].isna()] 
    datos_dentro_modelo = df[df[columna].notna()]
    return datos_dentro_modelo

def string_float(df):
    df['Lat'].astype(float)
    df['Long'].astype(float)
    df['altura'].astype(float)
    df['Gravedad_Obsrevada'].astype(float)
    df['Lat_Magna'].astype(float)
    df['Long_Magna'].astype(float)
    df['Grav_igsn71'].astype(float)

    
def interpolado_MODELOS(df,col_lat,col_lon,modelo_path,nombre_columna_modelo):
    df = df.reset_index(drop=True)
    coordenadas = list(zip(df[col_lon], df[col_lat]))
    interpolated_values_modelo = pygmt.grdtrack(points=coordenadas, grid=modelo_path, interpolation='l')
    df[nombre_columna_modelo] = interpolated_values_modelo[2]
    print(len(df))
    df = datos_Null(df, columna=nombre_columna_modelo)
    print(len(df))
    # string_float(df)
    return df
    
def diferencia(df, col_modelo, col_casilla, nombre_casilla):
    df[nombre_casilla] = df[col_modelo] - df[col_casilla]
    df['tipo_altura'] = np.nan  # Inicializar la columna
    
    # Aplicar la condición a cada fila usando .loc[]
    df.loc[(df[nombre_casilla] <= 2) & (df[nombre_casilla] >= -2), 'tipo_altura'] = 1
    df.loc[(df[nombre_casilla] > 2) | (df[nombre_casilla] < -2), 'tipo_altura'] = 0
    return df

def condicion_alturas(df):
    df_primer_grupo = df[df['altura'] >= 2000] 
    df_segundo_grupo = df[df['altura'] < 2000] 
    
    return df_primer_grupo,df_segundo_grupo

def condicion_3sigmas(df):
    media = df['dif_SRTM'].mean()
    #mediana = df['dif_SRTM'].median()
    desviacion_estandar = df['dif_SRTM'].std()
    df_filtrado = df[(df['dif_SRTM'] > -(media+3*(desviacion_estandar))) & (df['dif_SRTM'] < media+3*(desviacion_estandar))]
    df_datos_eliminados = df[(df['dif_SRTM'] < -(media+3*(desviacion_estandar))) | (df['dif_SRTM'] > media+3*(desviacion_estandar))]
    return df_filtrado,df_datos_eliminados

def Alturas(df):
    diferencia(df, col_modelo='SRTM30', col_casilla='altura', nombre_casilla='dif_SRTM')
    df_primer_grupo,df_segundo_grupo=condicion_alturas(df) #Aqui simplemente se clasifican según las alturas
    print('Observaciones en >=2000: ',len(df_primer_grupo))
    print('Observaciones en <2000: ',len(df_segundo_grupo))
    df_pg_filtrado,df_pg_eliminado=condicion_3sigmas(df_primer_grupo) #Aqui se realiza la regla de los tres sigmas
    print('Filtrados del primer Grupo : ',len(df_pg_filtrado))
    print('Eliminados del primer Grupo : ',len(df_pg_eliminado))
    df_sg_filtrado,df_sg_eliminado=condicion_3sigmas(df_segundo_grupo)
    print('Filtrados del Segundo Grupo : ',len(df_sg_filtrado))
    print('Eliminados del Segundo Grupo : ',len(df_sg_eliminado))
    print('Filtrados total: ',len(df_pg_filtrado)+len(df_sg_filtrado))
    print('Eliminados total: ',len(df_pg_eliminado)+len(df_sg_eliminado))
    dfs_filtrados = [df_pg_filtrado, df_sg_filtrado]
    df_concatenado_filtrado = pd.concat(dfs_filtrados, ignore_index=True)
    # CONCATENADO DE LOS DATA FRAMES ELIMINADOS
    dfs_eliminados = [df_pg_eliminado, df_sg_eliminado]
    df_concatenado_eliminado = pd.concat(dfs_eliminados, ignore_index=True)
    return df_concatenado_filtrado

def gravedad_teorica_somigliana(lat):
    global a, b, gamma_a, gamma_b
    # Fórmula corregida para evitar errores en el denominador
    gamma_0 = ((a * gamma_a * (np.cos(np.radians(lat)))**2 + b * gamma_b * (np.sin(np.radians(lat)))**2) /
               np.sqrt(a**2 * (np.cos(np.radians(lat)))**2 + b**2 * (np.sin(np.radians(lat)))**2))
    return gamma_0

def gamma_h(lat,gamma_0,h):
    global gamma_a,a,f,m
    gamma_h=gamma_0*(1-((2/a)*(1+f+m-(2*f*(np.sin(np.radians(lat)))**2))*h)+((3/(a**2))*(h**2)))
    return gamma_h

def calculo_perturbaciones(df):
    global gamma_a,gamma_b,b,a,f,m
    gravedades = np.array(df['Grav_igsn71'],dtype=float)
    latitud = np.array(df['Lat_Magna'],dtype=float)
    altura= np.array(df['altura'],dtype=float)
    gravedad_normal_p=np.zeros_like(altura)
    tipo_altura=np.array(df['tipo_altura'],dtype=float)
    sam=np.array(df['SAM'],dtype=float)
    gravedad_teorica=gravedad_teorica_somigliana(latitud)
    for i in range(len(df)):
        if tipo_altura[i]==1:#va con Don SAM
            h = altura[i]+sam[i]
            gravedad_normal_p[i]=gamma_h(latitud[i], gravedad_teorica[i], h)
        else:
            gravedad_normal_p[i]=gamma_h(latitud[i],gravedad_teorica[i],altura[i])
    perturbacion_gravedad=gravedades-gravedad_normal_p
    df['gravedad_teorica']=gravedad_teorica
    df['gravedad_normal_p']=gravedad_normal_p
    df['perturbacion_gravedad']=perturbacion_gravedad
    return df

