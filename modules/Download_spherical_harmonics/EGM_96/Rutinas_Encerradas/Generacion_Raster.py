import numpy as np
import rasterio
from rasterio.transform import from_origin
from rasterio.crs import CRS
def generacion_raster(puntos, col_lat, col_long,
                      tam_pixel, nombre_raster, nombre_col):

    # Preparar ráster
    pixel_size = tam_pixel/60
    latitudes = np.sort(puntos[col_lat].unique())[::-1]
    longitudes = np.sort(puntos[col_long].unique())
    raster_data = np.zeros((len(latitudes), len(longitudes)))

    # Asignar valores
    lat_idx_map = {lat: i for i, lat in enumerate(latitudes)}
    lon_idx_map = {lon: j for j, lon in enumerate(longitudes)}
    for _, row in puntos.iterrows():
        i = lat_idx_map[row[col_lat]]
        j = lon_idx_map[row[col_long]]
        raster_data[i, j] = row[nombre_col]

    # Transformación y escritura GeoTIFF
    transform = from_origin(longitudes[0] - pixel_size/2,
                            latitudes[0] + pixel_size/2,
                            pixel_size, pixel_size)
    # crs = CRS.from_wkt(
    #     "GEOGCS[\"GRS80\",DATUM[\"Geodetic_Reference_System_1980\","
    #     "SPHEROID[\"GRS 1980\",6378137,298.257222101]],PRIMEM[\"Greenwich\",0],"
    #     "UNIT[\"degree\",0.0174532925199433]]"
    # )
    crs='EPSG:4686'
    with rasterio.open(
        nombre_raster,
        'w', driver='GTiff',
        height=raster_data.shape[0], width=raster_data.shape[1],
        count=1, dtype=raster_data.dtype,
        crs=crs, transform=transform
    ) as dst:
        dst.write(raster_data, 1)
    print(f"Ráster generado exitosamente: {nombre_raster}")