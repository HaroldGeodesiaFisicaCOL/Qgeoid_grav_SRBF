import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d

# Parámetro del filtro gaussiano
sigma = 10  # Ajusta según sea necesario

# Leer archivo de datos
def procesar_archivo_txt(ruta_archivo):
    df = pd.read_csv(ruta_archivo, sep=',', decimal='.',index_col='Unnamed: 0')
    df = df.reset_index(drop=True)
    
    # Asegurar que las columnas necesarias están en el archivo
    columnas_necesarias = ["Proyecto", "Numero_Linea", "Pert_Grav"]
    if not all(col in df.columns for col in columnas_necesarias):
        raise ValueError("El archivo no contiene todas las columnas necesarias")
    
    # Procesar cada línea de vuelo de manera independiente
    resultados = []
    for (proyecto, linea), datos in df.groupby(["Proyecto", "Numero_Linea"]):
        X = np.linspace(0, len(datos) * 1000, len(datos))  # Aproximar distancia en metros
        g = datos["Pert_Grav"].values
        
        # Aplicar filtro gaussiano
        g_filtered = gaussian_filter1d(g, sigma=sigma)
        
        # Calcular la diferencia entre la perturbación y la perturbación filtrada
        diferencia = g - g_filtered
        
        # Agregar las columnas al DataFrame original
        datos["Pert_Grav_Filt"] = g_filtered
        datos["D_PGrav_PGraVF"] = diferencia
        
        resultados.append(datos)
        """
        # Crear carpeta para el proyecto si no existe
        carpeta_proyecto = os.path.join("resultadosFiltro1d", str(proyecto))
        os.makedirs(carpeta_proyecto, exist_ok=True)
        
        # ---- Mover la generación de gráficos aquí, fuera del bucle de puntos ----
        plt.figure(figsize=(10, 5))  # Ajustar tamaño si es necesario
        plt.plot(X, g, label='Original', marker='o', markersize=3)
        plt.plot(X, g_filtered, label='Filtrado (Gaussiano)', linestyle='--')
        plt.legend()
        plt.xlabel("Distancia (m)")
        plt.ylabel("Gravedad (mGal)")
        plt.title(f"Filtro Gaussiano - Proyecto {proyecto}, Línea {linea}")
        
        # Guardar la imagen
        ruta_grafico = os.path.join(carpeta_proyecto, f"Linea_{linea}.png")
        plt.savefig(ruta_grafico)
        plt.close()
        """
    
    # Combinar todos los resultados y guardar en el mismo archivo
    df_final = pd.concat(resultados)
    # df_final.to_csv(ruta_archivo_final, index=False, sep='\t')
    print(f"Proceso completado")
    return df_final
    

# Llamar a la función con la ruta del archivo TXT
# procesar_archivo_txt("Aero_grav_obs_Sin_NEXEN.txt","Aero_grav_obs_Sin_NEXEN_Remuestreado_2km.txt")
# df = pd.read_csv("Aero_grav_obs.txt", sep='\t', decimal='.')
# df_medellín_filtrado = df[
#     (df['Latitud'] <= 9.099463) & (df['Latitud'] >= 3.299463) &
#     (df['Longitud'] >= -78.478873) & (df['Longitud'] <= -72.678873)]
# df_medellín_filtrado.to_csv('Aero_grav_obs_IHRF.txt', index=False, sep='\t')
