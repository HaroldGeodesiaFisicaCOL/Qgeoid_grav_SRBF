import numpy as np
import pandas as pd

def generacion_pixeles(Lat_max, Lat_min, Lon_max, Lon_min,
                       tamano_pixel_min, ruta_guardado):

    # Tamaño del píxel en grados (por ejemplo, 1 minuto = 1/60°)
    paso_grados = tamano_pixel_min / 60.0

    # Número de filas y columnas
    n_filas = int(round((Lat_max - Lat_min) / paso_grados))
    n_columnas = int(round((Lon_max - Lon_min) / paso_grados))

    
    latitudes = Lat_min + (np.arange(n_filas) + 0.5) * paso_grados
    longitudes = Lon_min + (np.arange(n_columnas) + 0.5) * paso_grados

    # Generar pares (Latitud, Longitud)
    lon_grid, lat_grid = np.meshgrid(longitudes, latitudes)
    df_puntos = pd.DataFrame({
        "Latitud": lat_grid.ravel(),
        "Longitud": lon_grid.ravel()})
    
    print(f"Número de filas    : {n_filas}")
    print(f"Número de columnas : {n_columnas}")
    print(f"Paso utilizado     : {paso_grados:.20f}°")
    print(f"Total de puntos    : {len(df_puntos)}")
    
    # Guardar archivo
    df_puntos.to_csv(ruta_guardado, sep="\t", index=False, float_format="%.20f")
    

if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser(
        description="Generación de los centros de píxel del modelo Raster Qgeoide"
    )

    parser.add_argument("Lat_max", type=float, help="Latitud Máxima")
    parser.add_argument("Lat_min", type=float, help="Latitud Mínima")
    parser.add_argument("Lon_max", type=float, help="Longitud Máxima")
    parser.add_argument("Lon_min", type=float, help="Longitud Mínima")
    parser.add_argument("tamano_pixel", type=float, help="Tamaño del pixel en minutos")
    parser.add_argument("ruta_guardado", type=str, help="Ruta para guardar el archivo")

    args = parser.parse_args()

    generacion_pixeles(
        Lat_max=args.Lat_max,
        Lat_min=args.Lat_min,
        Lon_max=args.Lon_max,
        Lon_min=args.Lon_min,
        tamano_pixel_min=args.tamano_pixel,
        ruta_guardado=args.ruta_guardado
    )