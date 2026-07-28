import os
import pandas as pd
import numpy as np


R_EARTH = 6378137

def haversine(lon1, lat1, lon2, lat2):
    """Distancia en metros entre dos lon/lat en grados decimales."""
    lon1, lat1, lon2, lat2 = map(np.radians, (lon1, lat1, lon2, lat2))
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2
    return 2 * R_EARTH * np.arcsin(np.sqrt(a))

def sample_by_distance(df, sample_distance, output_file, report_file):
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Lectura de Datos
    # df = pd.read_csv(input_file, sep="\t", dtype=str)
    
    # Copiar 
    lat_text = df["Latitud"].copy()
    lon_text = df["Longitud"].copy()
    
    # Conversión a numérico para cálculos
    df["Latitud"]  = pd.to_numeric(df["Latitud"], errors="coerce")
    df["Longitud"] = pd.to_numeric(df["Longitud"], errors="coerce")
    if "Grav_Obs_C" in df.columns:
        df["Grav_Obs_C"] = pd.to_numeric(df["Grav_Obs_C"], errors="coerce")
    df = df.dropna(subset=["Latitud", "Longitud"])
    
    sampled_idx = []  # índices del DF original
    informe = {}

    # 2) Muestreo
    for (proj, lin), group in df.groupby(["Proyecto", "Numero_Linea"], sort=False):
       
        orig_idx = group.index.to_numpy()
        lons  = group["Longitud"].values
        lats  = group["Latitud"].values
        
        # Acumulando distancias entre puntos.
        if len(group) > 1:
            dists = haversine(lons[:-1], lats[:-1], lons[1:], lats[1:])
            cumdist = np.concatenate([[0], np.cumsum(dists)])
        else:
            dists = np.array([])
            cumdist = np.array([0])
        
        
        proj_info = informe.setdefault(proj, {
            "total_dist_sum": 0.0,
            "count_dist": 0,
            "total_original_pts": 0,
            "total_sampled_pts": 0
        })
        proj_info["total_dist_sum"]  += dists.sum()
        proj_info["count_dist"]      += len(dists)
        proj_info["total_original_pts"] += len(group)
        
        # Targets de muestreo
        maxd    = cumdist[-1]
        targets = np.arange(0, maxd + sample_distance, sample_distance)
        
        # Selección secuencial (evita retrocesos)
        last_idx_pos = -1
        for t in targets:
            i = np.argmin(np.abs(cumdist - t))
            if i <= last_idx_pos:
                i = last_idx_pos + 1
                if i >= len(group):
                    break
            sampled_idx.append(orig_idx[i])  # usamos índice real del df
            last_idx_pos = i
        
        proj_info["total_sampled_pts"] += len(targets)
    
    # 3) Extraer puntos muestreados
    sampled_df = df.loc[sampled_idx].reset_index(drop=True).copy()
    
    # Restaurar coordenadas originales
    sampled_df["Latitud"]  = lat_text.loc[sampled_idx].values
    sampled_df["Longitud"] = lon_text.loc[sampled_idx].values
    
    # Guardar en TXT por tabulación
    sampled_df.to_csv(output_file, sep="\t", index=False)
    
    # 4) Informe
    lines = ["Informe de muestreo por proyecto\n",
             "Proyecto\tDistancia promedio (m)\tPuntos muestreados por cada 100 originales\n"]
    
    for proj, info in informe.items():
        avg_dist = info["total_dist_sum"] / info["count_dist"] if info["count_dist"] > 0 else 0
        ratio    = info["total_sampled_pts"] / info["total_original_pts"] * 100 if info["total_original_pts"] > 0 else 0
        lines.append(f"{proj}\t{avg_dist:.2f}\t{ratio:.1f}\n")
    
    with open(report_file, "w") as f:
        f.writelines(lines)
    
    # Mostrar en consola
    print("\n".join(lines))
    print(f"\nArchivo de muestreo guardado en: {output_file}")
    print(f"Informe guardado en: {report_file}")
    
    return sampled_df


# if __name__ == "__main__":
#     INPUT        = "C:/Users/carlos.rico/Desktop/Correccion_de_Sesgos/3_Observaciones/Datos_IHRF.txt"
#     OUTPUT       = "C:/Users/carlos.rico/Desktop/Correccion_de_Sesgos/3_Observaciones/Aero_IHRF_M.txt"
#     REPORT       = "C:/Users/carlos.rico/Desktop/Correccion_de_Sesgos/Informe_Muestreo.txt"
#     DIST_METROS  = 1000
    
#     muestreados = sample_by_distance(INPUT, DIST_METROS, OUTPUT, REPORT)
