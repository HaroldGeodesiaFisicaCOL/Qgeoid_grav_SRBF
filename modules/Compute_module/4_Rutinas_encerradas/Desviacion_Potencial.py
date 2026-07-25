import pygmt
import geopandas as gpd
import rasterio
import pandas as pd
import numpy as np
import rasterio
from rasterio.transform import from_origin
from rasterio.crs import CRS


def mapa_local(ruta_qgeoide_tiff,ruta_mapa_pdf):
    # Ruta al archivo GeoTIFF
    tiff_path = ruta_qgeoide_tiff  # Cambia esta ruta al archivo GeoTIFF
    
    # Obtener la extensión del raster
    info = pygmt.grdinfo(tiff_path, per_column=True).split()  # Divide la salida en valores individuales
    region = [float(info[0]), float(info[1]), float(info[2]), float(info[3])]  # xmin, xmax, ymin, ymax
    
    # Crear la figura
    fig = pygmt.Figure()
    
    # Leer el archivo TIFF y mostrarlo con una paleta personalizada
    fig.grdimage(
        grid=tiff_path,
        region=region,  # Ajusta la región al raster
        projection="M15c",  # Proyección Mercator
        frame="af",          # Marco automático
        cmap="jet",         # Paleta de colores azul-verde-amarillo-rojo
        # shading=True        # Opcional: agrega sombreado para realzar el relieve
    )
    
    # Agregar la barra de colores
    fig.colorbar(frame=["a1", "x+lPotencial Perturbador", "y+lm2/s2"])
    
    # Agregar líneas de delimitación de los países
    fig.coast(
        region=region,      # Asegura que la región sea la misma que el raster
        borders="1/0.5p",
        shorelines="1/0.5p",
        frame="af",
    )
    
    # Guardar el mapa como PDF
    fig.savefig(ruta_mapa_pdf)
    fig.savefig(ruta_mapa_pdf+'.png')
    
def mapa_local_std(ruta_qgeoide_tiff, ruta_mapa_pdf):

    info = pygmt.grdinfo(ruta_qgeoide_tiff, per_column=True).split()

    region = [float(info[0]), float(info[1]),
              float(info[2]), float(info[3])]
    
    zmin = float(info[4])
    zmax = float(info[5])
    
    fig = pygmt.Figure()
    
    pygmt.makecpt(cmap="jet", series=[zmin-1e-2, zmax+1e-2])
    
    fig.grdimage(
        grid=ruta_qgeoide_tiff,
        region=region,
        projection="M15c",
        frame="af",
        interpolation="n",   # evita huecos por interpolación
        nan_transparent=False
    )
    
    fig.colorbar(frame=["a", "x+lDesviación Estándar", "y+lm2/s2"])
    
    fig.coast(
        region=region,
        borders="1/0.5p",
        shorelines="1/0.5p",
    )
    # Guardar el mapa como PDF
    fig.savefig(ruta_mapa_pdf)
    fig.savefig(ruta_mapa_pdf+'.png')

    
def generar_raster(df, output_filename):
    """
    Genera raster correctamente orientado:
    - Norte → Sur
    - Oeste → Este
    - Resolución calculada automáticamente
    """

    if not {'Latitud', 'Longitud', 'Std_Zeta'}.issubset(df.columns):
        raise ValueError("El dataframe debe contener Latitud, Longitud y Std_T")

    # --------------------------------------------------
    # 1. Orden espacial explícito
    # --------------------------------------------------
    df_sorted = df.sort_values(['Latitud', 'Longitud'], ascending=[False, True])

    # --------------------------------------------------
    # 2. Pivot espacial correcto
    # --------------------------------------------------
    grid = df_sorted.pivot(index='Latitud',
                           columns='Longitud',
                           values='Std_T')

    # Forzar orden correcto
    grid = grid.sort_index(ascending=False)   # Norte → Sur
    grid = grid.sort_index(axis=1)            # Oeste → Este

    raster_data = grid.values.astype(np.float64)

    latitudes = grid.index.values
    longitudes = grid.columns.values

    nrows, ncols = raster_data.shape

    if nrows < 2 or ncols < 2:
        raise ValueError("No hay suficientes puntos para construir raster")

    # --------------------------------------------------
    # 3. Calcular resolución real
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
    print(f"Resolución lat: {lat_step*60:.20f} minutos")
    print(f"Resolución lon: {lon_step*60:.20f} minutos")
    print("--------------------------------------------------")

def principal(raster_puntos_centrales,ruta_qgeoide_tiff1, ruta_mapa_pdf1,output_filename,pixel_size,ruta_qgeoide_tiff2,ruta_mapa_pdf2):
    mapa_local(ruta_qgeoide_tiff1, ruta_mapa_pdf=ruta_mapa_pdf1)
    df = pd.read_csv(raster_puntos_centrales,delimiter='\t')
    generar_raster(df=df, output_filename = output_filename)
    mapa_local_std(ruta_qgeoide_tiff=ruta_qgeoide_tiff2, ruta_mapa_pdf=ruta_mapa_pdf2)


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("ruta_raster_centrales", type=str, help="Ruta del txt del raster puntos centrales")
    parser.add_argument("ruta_qgeoide_tiff1", type=str, help="Ruta del modelo de potencial Tiff")
    parser.add_argument("ruta_mapa_pdf1", type=str, help="Ruta para el mapa del potencial T")
    parser.add_argument("output_filename", type=str, help="Ruta para raster del STD")
    parser.add_argument("pixel_size", type=float, help="Tamaño de pixel del STD del potencial")
    parser.add_argument("ruta_mapa_pdf2", type=str, help="Ruta para el mapa del STD del potencial T")
    
    args = parser.parse_args()
    
    principal(raster_puntos_centrales=args.ruta_raster_centrales, 
              ruta_qgeoide_tiff1=args.ruta_qgeoide_tiff1, 
              ruta_mapa_pdf1=args.ruta_mapa_pdf1, 
              output_filename=args.output_filename, 
              pixel_size=args.pixel_size, 
              ruta_qgeoide_tiff2=args.output_filename, 
              ruta_mapa_pdf2=args.ruta_mapa_pdf2)