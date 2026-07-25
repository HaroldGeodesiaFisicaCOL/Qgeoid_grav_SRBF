import numpy as np
import pandas as pd
from tqdm import tqdm 

def generador_Reuter_grid(n_max):
    # Parámetros del modelo
    gr = 6378137.0  # Radio ecuatorial de la Tierra (GRS80)
    #n_max = 2189 # Valor de n_max definido  
    gamma = n_max + 1
    
    # Calcular la cantidad total de puntos Z según la ecuación (5.4)
    Z = (n_max + 1)**2
    print(f"Número total de puntos en la esfera: {Z}")
    
    # Generar la Reuter Grid
    latitudes = []
    longitudes = []
    
    for l in tqdm(range(1, gamma)):
        delta_phi = np.pi / gamma
        phi_l = -np.pi / 2 + l * delta_phi
        num_m = int(np.floor((2 * np.pi) / np.arccos((np.cos(delta_phi) - np.sin(phi_l)**2) / np.cos(phi_l)**2)))
        delta_lambda = 2 * np.pi / num_m
        for m in range(num_m):
            lambda_m = m * delta_lambda
            lat = np.degrees(phi_l)
            lon = np.degrees(lambda_m)
            latitudes.append(lat)
            longitudes.append(lon)
    
    # Convertir listas a arrays numpy
    latitudes = np.array(latitudes)
    longitudes = np.array(longitudes)
    
    # Guardar resultados en un único archivo TXT con 6 cifras decimales
    data = np.column_stack((latitudes, longitudes))
    df = pd.DataFrame(data, columns=["Latitud", "Longitud"])
    #np.savetxt(ruta_archivo_reuter, data, delimiter="\t", fmt="%.6f", header="Latitud\tLongitud", comments="")
    return df 

def filtrar_datos(df, archivo_salida, Lat_max,Lat_min,Lon_max,Lon_min):
    # Convertir longitudes mayores a 180 a su equivalente negativo
    df['Longitud'] = df['Longitud'].apply(lambda x: x - 360 if x > 180 else x)
    abajo=Lat_min
    arriba=Lat_max
    izquierda=Lon_min
    derecha=Lon_max
    # Filtrar las filas según las coordenadas especificadas
    filtro = (
        (df['Latitud'] >= abajo) & (df['Latitud'] <= arriba) &
        (df['Longitud'] >= izquierda) & (df['Longitud'] <= derecha)
    )
    df_filtrado = df[filtro]
    print(f"Número de filas filtradas: {len(df_filtrado)}")

    # Guardar el archivo filtrado
    df_filtrado.to_csv(archivo_salida, sep='\t', index=False)
    print(f"Archivo filtrado guardado en: {archivo_salida}")
    
if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('n_max',type=int,help='Número de frecuencias máxima del modelo de Qgeoide Gravimétrico')
    parser.add_argument('ruta_archivo_reuter_local',type=str,help='Ruta donde se almacenará la reuter local')
    parser.add_argument('Lat_max',type=float,help='Latitud Máxima')
    parser.add_argument('Lat_min',type=float,help='Latitud Mínima')
    parser.add_argument('Lon_max',type=float,help='Longitud Máxima')
    parser.add_argument('Lon_min',type=float,help='Longitud Mínima')
    
    args = parser.parse_args()
    
    data = generador_Reuter_grid(n_max = args.n_max)
    
    filtrar_datos(data, 
                  archivo_salida=args.ruta_archivo_reuter_local,
                  Lat_max=args.Lat_max, Lat_min=args.Lat_min, Lon_max=args.Lon_max,Lon_min=args.Lon_min)