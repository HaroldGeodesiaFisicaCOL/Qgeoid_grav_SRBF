import numpy as np
from scipy.stats import t 
import pandas as pd
from shapely.geometry import Point
import geopandas as gpd
import pygmt 
import matplotlib.pyplot as plt
from scipy.stats import norm
import os

def significancia_1cola(archivo_parametros,ruta_listado,archivo_varcov,archivo_estadistico_t,significancia_param):
      
    alpha = 0.10  
    
    # === CARGA DE DATOS ===
    parametros = np.loadtxt(archivo_parametros)  # Vector 1D
    num_parametros = parametros.shape
    num_parametros = int(num_parametros[0])
    ## se debe de cargar las observaciones
    listado = pd.read_csv(ruta_listado,sep='\t',header=None)
    numero_observaciones = np.sum([len(pd.read_csv(y,sep='\t')) for y in listado[1]])
    numero_observaciones  = int(numero_observaciones )
    
    var_cov = np.load(archivo_varcov)         # Matriz cuadrada
    var_cov = var_cov 
    grados_de_libertad = numero_observaciones - num_parametros
    # === CÁLCULOS ===
    # Errores estándar de los coeficientes
    errores_std = np.sqrt(np.diag(var_cov))
    
    # Estadístico t (valor absoluto)
    t_stats = np.abs(parametros) / errores_std
    
    # Valor crítico para prueba unilateral
    t_crit = t.ppf(1 - alpha, df=grados_de_libertad)
    
    # Determinar significancia
    significancia = t_stats > t_crit
    
    # Contar cuántos parámetros son significativos y cuántos no
    num_significativos = np.sum(significancia)
    num_no_significativos = len(significancia) - num_significativos
    
    # === GUARDAR RESULTADOS ===
    # Guardar t-stats
    np.savetxt(archivo_estadistico_t, t_stats, fmt='%.6f')
    
    # Guardar resultado de significancia
    with open(significancia_param, 'w') as f:
        for i, sig in enumerate(significancia):
            f.write(f'Parametro {i+1}: {"Significativo" if sig else "No significativo"}\n')
    
    # === RESUMEN ===
    print("Prueba t completada (unilateral con valores absolutos).")
    print(f"- Grados de libertad: {grados_de_libertad}")
    print(f"- Nivel de significancia (alpha): {alpha}")
    print(f"- Valor crítico t unilateral: {t_crit:.4f}")
    print("- Resultados guardados en:")
    print("   • t_stats_resultados.txt")
    print("   • significancia_resultados.txt")
    print("\nResumen de resultados:")
    print(f"   • Parámetros significativos: {num_significativos}")
    print(f"   • Parámetros no significativos: {num_no_significativos}")
    return t_stats,t_crit,parametros
    
def lectura_reuter(ruta_reuter):
    reuter = pd.read_csv(ruta_reuter, sep='\t', header=0)
    return reuter

def asignacion(df,nombre_col,estadistico,nombre_param,parametros):
    df[nombre_col]=estadistico
    df[nombre_param] = parametros

def pandastogeopandas(df):
    geometry = [Point(lon, lat) for lon, lat in zip(df['Longitud'], df['Latitud'])]
    gdf = gpd.GeoDataFrame(df, geometry=geometry)
    gdf.set_crs('EPSG:4686', allow_override=True, inplace=True)
    return gdf

def mapa_gmt(df, ruta_mapa):
    minx, miny, maxx, maxy = df.total_bounds
    region = [minx - 0.5, maxx + 0.5, miny - 0.5, maxy + 0.5]
    
    t_clip_max = np.percentile(df["t_stats"], 95)
    df["t_plot"] = df["t_stats"].clip(lower=1 ,upper=t_clip_max)
    
    
    zmax = t_clip_max + 1e-6

    # Crear paleta de colores temporal
    pygmt.makecpt(cmap="jet", series=[1, zmax], continuous=True, output="cpt_mapa_gmt.cpt")

    fig = pygmt.Figure()
    fig.coast(
        region=region,
        shorelines='1/0.05p',
        projection="M15c",
        land="lightgray",
        # water="lightblue",
        borders="1/0.5p,black",
        frame="af",
    )
    fig.plot(
        x=df["Longitud"],
        y=df["Latitud"],
        fill=df["t_plot"],
        style="c0.05c",
        cmap="cpt_mapa_gmt.cpt",
    )
    fig.colorbar(frame='af+lEstadístico t', cmap="cpt_mapa_gmt.cpt")
    fig.savefig(ruta_mapa, dpi=600)
    fig.savefig(ruta_mapa+'.png')
    #fig.show()

    

def mapa_parametros(df, ruta_mapa):
    minx, miny, maxx, maxy = df.total_bounds
    region = [minx - 0.5, maxx + 0.5, miny - 0.5, maxy + 0.5]

    zmax = max(abs(df["param"].min()), abs(df["param"].max()))
    zmin = -zmax

    # pygmt.makecpt(
    #     cmap="jet",
    #     series=[zmin-1e-11, zmax+1e-11, (zmax - zmin) / 100],
    #     truncate=[0, 1],
    #     continuous=True,
    #     output="cpt_parametros.cpt"
    # )
    
    pygmt.makecpt(
        cmap="jet",
        series=[-2e-5, 2e-5, (2e-5 + 2e-5) / 100],
        truncate=[0, 1],
        continuous=True,
        output="cpt_parametros.cpt"
    )

    fig = pygmt.Figure()
    fig.coast(
        region=region,
        projection="M15c",
        shorelines="1/0.05p",
        land="lightgray",
        # water="lightblue",
        borders="1/0.5p,black",
        frame="af"
    )

    df = df.copy()
    # df["param_clip"] = df["param"].clip(lower=zmin, upper=zmax)
    df["param_clip"] = df["param"].clip(lower=-2e-5, upper=2e-5)

    fig.plot(
        x=df["Longitud"],
        y=df["Latitud"],
        fill=df["param_clip"],
        style="c0.07c",
        cmap="cpt_parametros.cpt",
    )

    fig.colorbar(frame='xafg+lParámetro Estimado (m²/s²)', cmap="cpt_parametros.cpt")
    fig.savefig(ruta_mapa)
    fig.savefig(ruta_mapa+'.png')
    #fig.show()


def mapa_parametros_STD(df, ruta_mapa):
    minx, miny, maxx, maxy = df.total_bounds
    region = [minx - 0.5, maxx + 0.5, miny - 0.5, maxy + 0.5]
    zmin = df["STD_Param"].min()
    zmax = df["STD_Param"].max()

    pygmt.makecpt(cmap="jet", series=[zmin-1e-11, zmax+1e-11], continuous=True, output="cpt_std.cpt")

    fig = pygmt.Figure()
    fig.coast(
        region=region,
        shorelines='1/0.05p',
        projection="M15c",
        land="lightgray",
        # water="lightblue",
        borders="1/0.5p,black",
        frame="af",
    )
    fig.plot(
        x=df["Longitud"],
        y=df["Latitud"],
        fill=df['STD_Param'],
        style="c0.05c",
        cmap="cpt_std.cpt",
    )
    fig.colorbar(frame='af+lDesviación Estándar (m²/s²)', cmap="cpt_std.cpt")
    fig.savefig(ruta_mapa, dpi=600)
    fig.savefig(ruta_mapa+'.png')
    #fig.show()
    
    
def parametros_sign(df, t_crit, ruta_mapa):
    df_filtrado = df[df["t_stats"] >= t_crit].copy()
    t_clip_max = np.percentile(df_filtrado["t_stats"], 95)
    df_filtrado["t_plot"] = df_filtrado["t_stats"].clip(lower=t_crit ,upper=t_clip_max)
    minx, miny, maxx, maxy = df_filtrado.total_bounds
    region = [minx - 0.5, maxx + 0.5, miny - 0.5, maxy + 0.5]
 
    zmin = t_crit
    zmax = t_clip_max + 1e-6
 
    pygmt.makecpt(
        cmap="jet",
        series=[zmin, zmax],
        continuous=True,
        output="cpt_significativos.cpt"
    )
 
    fig = pygmt.Figure()
    fig.coast(
        region=region,
        shorelines='1/0.05p',
        projection="M15c",
        land="lightgray",
        # water="lightblue",
        borders="1/0.5p,black",
        frame="af"
    )
 
    fig.plot(
        x=df_filtrado["Longitud"],
        y=df_filtrado["Latitud"],
        fill=df_filtrado["t_plot"],  
        style="c0.05c",
        cmap="cpt_significativos.cpt"
    )
 
    fig.colorbar(frame='af+lEstadístico t', cmap="cpt_significativos.cpt")
    fig.savefig(ruta_mapa, dpi=600)
    fig.savefig(ruta_mapa+'.png')


"""
REVISIÓN DE LAS ESTADÍSITCAS BÁSICAS DE LOS DATOS DE OBSERVACIONES TERRESTRES

"""
plt.rcParams['text.usetex'] = True 
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['cm']
plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'


def estadisticos(df, columnas, labels,ruta_ploteo):
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
        count, bins, ignored = plt.hist(datos, bins=60, color='skyblue', edgecolor='black', density=True)

        # Curva normal superpuesta
        x = np.linspace(min(bins), max(bins), 1000)
        #plt.plot(x, norm.pdf(x, media, std), 'r-', lw=2, label='Curva normal')

        plt.xlabel(label)
        plt.ylabel('Densidad')
        plt.grid(True)
        plt.tight_layout()

        # Guardar el plot usando el nombre de la columna
        plt.savefig(ruta_ploteo)
        print(f'Guardado: {ruta_ploteo}')
        plt.show()
        
def estadisticos2(df, columnas, labels,ruta_ploteo):
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
        count, bins, ignored = plt.hist(datos, bins=60, color='skyblue', edgecolor='black', density=False)

        plt.xlabel(label)
        plt.ylabel('Densidad')
        plt.grid(True)
        plt.tight_layout()

        # Guardar el plot usando el nombre de la columna
        plt.savefig(ruta_ploteo)
        print(f'Guardado: {ruta_ploteo}')
        plt.show()

# Funcion de ejecución del script
def principal(archivo_parametros,ruta_listado,archivo_varcov,archivo_estadistico_t,histo_parametros,histo_estadisticot,significancia_param,ruta_reuter,Mapa_parametros_sign,Mapa_parametros_sign_solo,Mapa_parametros,
              Mapa_parametros_STD):

    t_stats,t_crit,parametros=significancia_1cola(archivo_parametros=archivo_parametros,ruta_listado=ruta_listado,archivo_varcov=archivo_varcov,archivo_estadistico_t=archivo_estadistico_t,
                                                  significancia_param=significancia_param)
    reuter = lectura_reuter(ruta_reuter=ruta_reuter)
    asignacion(reuter, nombre_col='t_stats', estadistico=t_stats,nombre_param='param',parametros=parametros)
    reuter.to_csv(ruta_reuter,sep='\t', index=False)
    reuter_espacial = pandastogeopandas(reuter)
    mapa_gmt(df=reuter_espacial,ruta_mapa=Mapa_parametros_sign)
    parametros_sign(df=reuter_espacial, t_crit=t_crit,ruta_mapa=Mapa_parametros_sign_solo)
    mapa_parametros(df=reuter_espacial, ruta_mapa=Mapa_parametros)
    arreglo= ['param']
    labels=[r'$m^2s^{-2}$']
    estadisticos(df = reuter_espacial, columnas=arreglo,labels=labels,ruta_ploteo=histo_parametros)
    arreglo= ['t_stats']
    labels=[r'$\hat{t}$']
    estadisticos2(df = reuter_espacial, columnas=arreglo,labels=labels,ruta_ploteo=histo_estadisticot)
    var_cov = np.load(archivo_varcov)
    errores_std = np.sqrt(np.diag(var_cov))
    reuter_espacial['STD_Param'] = errores_std
    mapa_parametros_STD(df=reuter_espacial, ruta_mapa= Mapa_parametros_STD)
    # os.remove('cpt_mapa_gmt.cpt')
    # os.remove('cpt_std.cpt')
    # os.remove('pt_significativos.cpt')

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument('archivo_parametros',type=str,help='Archivo txt con los parámetros')
    parser.add_argument('ruta_listado',type=str,help='Archivo txt con los el listado')
    parser.add_argument('archivo_varcov',type=str,help='Archivo de la matriz de varianza covarinaza')
    parser.add_argument('archivo_estadistico_t',type=str,help='Archivo donde quedarán los stadisticos t calculados')
    parser.add_argument('histo_parametros',type=str,help='Ploteo del histograma de los parámetros')
    parser.add_argument('histo_estadisticot',type=str,help='Ploteo del histograma de significancia de parámetros')
    parser.add_argument('significancia_param',type=str,help='Archivo donde quedarán los stadisticos t calculados')
    parser.add_argument('ruta_reuter',type=str,help='Archivo txt con la reuter Grid')
    parser.add_argument('Mapa_parametros_sign',type=str,help='Ruta par el mapa de los parámetros significativos')
    parser.add_argument('Mapa_parametros_sign_solo',type=str,help='Ruta para el mapa de los parámetros significativos solos')
    parser.add_argument('Mapa_parametros',type=str,help='Ruta para el mapa de los parámetros estimados')
    parser.add_argument('Mapa_parametros_STD',type=str,help='Ruta para el mapa de STD de los parámetros estimados')
    
    args = parser.parse_args()
    
    principal(archivo_parametros=args.archivo_parametros,
              ruta_listado=args.ruta_listado,
              archivo_varcov=args.archivo_varcov,
              archivo_estadistico_t=args.archivo_estadistico_t,
              histo_parametros =args.histo_parametros,
              histo_estadisticot = args.histo_estadisticot,
              significancia_param = args.significancia_param,
              ruta_reuter=args.ruta_reuter, 
              Mapa_parametros_sign=args.Mapa_parametros_sign, 
              Mapa_parametros_sign_solo=args.Mapa_parametros_sign_solo, 
              Mapa_parametros=args.Mapa_parametros,
              Mapa_parametros_STD=args.Mapa_parametros_STD)

