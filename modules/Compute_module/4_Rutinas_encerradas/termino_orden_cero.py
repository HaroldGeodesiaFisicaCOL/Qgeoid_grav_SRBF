import argparse
import pandas as pd
import numpy as np
import rasterio
from rasterio.transform import from_origin
from rasterio.crs import CRS


## CONSTANTES NECESARIAS PARA EL CÁLCULO
Uo = 62636860.850 
Wo = 62636853.4 
def lectura_txt(ruta):
    return pd.read_csv(ruta,sep='\t')
        
def generar_raster(df, output_filename):
    """
    Genera un GeoTIFF correctamente orientado
    usando pivot (N→S, W→E).
    No usa reshape ni loops manuales.
    """

    if not {'Latitud', 'Longitud', 'anom_res'}.issubset(df.columns):
        raise ValueError("El dataframe debe tener 'Latitud', 'Longitud' y 'anom_res'.")

    # --------------------------------------------------
    # 1. Ordenar explícitamente (CRÍTICO)
    # --------------------------------------------------
    df_sorted = df.sort_values(['Latitud', 'Longitud'], ascending=[False, True])

    # --------------------------------------------------
    # 2. Pivot espacial correcto
    # --------------------------------------------------
    grid = df_sorted.pivot(index='Latitud',
                           columns='Longitud',
                           values='zeta_final')

    # Asegurar orden espacial
    grid = grid.sort_index(ascending=False)  # Norte → Sur
    grid = grid.sort_index(axis=1)           # Oeste → Este

    raster_data = grid.values.astype(np.float64)

    latitudes = grid.index.values
    longitudes = grid.columns.values

    nrows, ncols = raster_data.shape

    # --------------------------------------------------
    # 3. Calcular tamaño REAL del pixel
    # --------------------------------------------------
    lat_step = np.abs(latitudes[0] - latitudes[1])
    lon_step = np.abs(longitudes[1] - longitudes[0])

    west = longitudes[0] - lon_step / 2
    north = latitudes[0] + lat_step / 2

    transform = from_origin(west, north, lon_step, lat_step)

    crs = CRS.from_epsg(4686)

    # --------------------------------------------------
    # 4. Escribir raster
    # --------------------------------------------------
    with rasterio.open(
        output_filename,
        'w',
        driver='GTiff',
        height=nrows,
        width=ncols,
        count=1,
        dtype='float64',
        crs=crs,
        transform=transform,
        nodata=np.nan,
        compress='lzw'
    ) as dst:
        dst.write(raster_data, 1)

    print("--------------------------------------------------")
    print(f"Ráster generado correctamente: {output_filename}")
    print(f"Filas: {nrows}")
    print(f"Columnas: {ncols}")
    print(f"Resolución lat: {lat_step*60:.10f} minutos")
    print(f"Resolución lon: {lon_step*60:.10f} minutos")
    print("--------------------------------------------------")        
        
def main_calculo_termino_orden_cero(ruta_puntos_centrales,ruta_final,tamano_pixel):
    raster = lectura_txt(ruta_puntos_centrales)
    raster['zeta_cero'] = (Wo - Uo) / (raster['grav_telu']/100000)
    raster['zeta_final'] = raster['anom_res'] - raster['zeta_cero'] 
    generar_raster(df=raster, output_filename=ruta_final)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Proceso Restore (adaptado)')
    parser.add_argument("ruta_puntos", type = str,help="TXT de puntos centrales")
    parser.add_argument("ruta_final_modelo", type = str, help="Ruta modelo Qgeoide")
    parser.add_argument("tam_pixel", type = float, help="Tamaño de pixel del modelo")
    args = parser.parse_args()
    
    main_calculo_termino_orden_cero(ruta_puntos_centrales=args.ruta_puntos, 
                                    ruta_final=args.ruta_final_modelo, 
                                    tamano_pixel=args.tam_pixel)
    
    