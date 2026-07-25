import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import from_origin
from rasterio.crs import CRS


def generacion_raster(ruta_matriz_diseno_pixel, ruta_parametros, raster_puntos_centrales,
                      nombre_raster, param_nom):

    # ----------------------------
    # 1. Cargar matriz de diseño
    # ----------------------------
    matriz_diseno = np.load(ruta_matriz_diseno_pixel)
    n_cols = matriz_diseno.shape[1]

    parametros = np.loadtxt(ruta_parametros, delimiter='\t')
    parametros = parametros.flatten() if parametros.ndim > 1 else parametros

    if parametros.size != n_cols:
        raise ValueError(
            f"Número de parámetros ({parametros.size}) "
            f"no coincide con columnas de matriz ({n_cols})"
        )

    resultado = matriz_diseno.dot(parametros)

    # ----------------------------
    # 2. Leer puntos centrales
    # ----------------------------
    puntos = pd.read_csv(raster_puntos_centrales, sep='\t')

    col_map = {'T': 'Potencial_Perturbador',
               'Z': 'Anomalia_altura',
               'Std': 'Std_Qgeoide'}

    nombre_col = col_map.get(param_nom, param_nom)
    puntos[nombre_col] = resultado
    puntos.to_csv(raster_puntos_centrales, sep='\t', index=False,float_format="%.20f")

    print(f"Archivo con resultados actualizado: {raster_puntos_centrales}")

    # ----------------------------
    # 3. Reconstruir grilla ordenada
    # ----------------------------
    puntos[nombre_col] = resultado

    # Ordenar explícitamente
    puntos = puntos.sort_values(['Latitud', 'Longitud'], ascending=[False, True])
    
    # Pivotear
    grid = puntos.pivot(index='Latitud',
                        columns='Longitud',
                        values=nombre_col)
    
    # Asegurar orden correcto
    grid = grid.sort_index(ascending=False)
    grid = grid.sort_index(axis=1)
    
    raster_data = grid.values
    
    latitudes = grid.index.values
    longitudes = grid.columns.values
    
    n_filas, n_columnas = raster_data.shape

    # ----------------------------
    # 4. Calcular tamaño REAL del pixel
    # ----------------------------
    lat_step = np.abs(latitudes[0] - latitudes[1])
    lon_step = np.abs(longitudes[1] - longitudes[0])

    # Coordenadas esquina superior izquierda
    west = longitudes[0] - lon_step / 2
    north = latitudes[0] + lat_step / 2

    transform = from_origin(west, north, lon_step, lat_step)

    crs = CRS.from_epsg(4686)

    # ----------------------------
    # 5. Escribir GeoTIFF
    # ----------------------------
    with rasterio.open(
        nombre_raster,
        'w',
        driver='GTiff',
        height=n_filas,
        width=n_columnas,
        count=1,
        dtype=raster_data.dtype,
        crs=crs,
        transform=transform
    ) as dst:
        dst.write(raster_data, 1)

    print("--------------------------------------------------")
    print(f"Ráster generado: {nombre_raster}")
    print(f"Filas: {n_filas}")
    print(f"Columnas: {n_columnas}")
    print(f"Resolución lat: {lat_step*60:.20f} minutos")
    print(f"Resolución lon: {lon_step*60:.20f} minutos")
    print("--------------------------------------------------")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Creación Raster')

    parser.add_argument('ruta_matriz_diseno_pixel', type=str)
    parser.add_argument('ruta_parametros', type=str)
    parser.add_argument('raster_puntos_centrales', type=str)
    parser.add_argument('nombre_raster', type=str)
    parser.add_argument('param_nom', type=str,
                        choices=['T', 'Z', 'Std'],
                        help='T, Z, Std')

    args = parser.parse_args()

    generacion_raster(
        args.ruta_matriz_diseno_pixel,
        args.ruta_parametros,
        args.raster_puntos_centrales,
        args.nombre_raster,
        args.param_nom
    )