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

def calculo_perturbaciones(ruta,col_grav,col_lat,col_h):
    global gamma_a,gamma_b,b,a,f,m
    df = pd.read_csv(ruta,sep='\t')
    gravedades = np.array(df[col_grav],dtype=float)
    latitud = np.array(df[col_lat],dtype=float)
    altura= np.array(df[col_h],dtype=float)
    gravedad_normal_p=np.zeros_like(altura)
    gravedad_teorica=gravedad_teorica_somigliana(latitud)
    for i in range(len(df)):
        gravedad_normal_p[i]=gamma_h(latitud[i],gravedad_teorica[i],altura[i])
    perturbacion_gravedad=gravedades-gravedad_normal_p
    df[col_grav]=perturbacion_gravedad
    df.to_csv(ruta,sep='\t',index=False)
