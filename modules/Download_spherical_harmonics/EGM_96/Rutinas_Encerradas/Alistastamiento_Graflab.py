import pandas as pd
import os 
import numpy as np
import rasterio
from rasterio.windows import Window
from rasterio.transform import xy
from pyproj import Transformer

def raster_a_txt_centros_por_bloques(
    raster_path: str,
    out_txt: str,
    dst_epsg: str = "EPSG:4326",
    incluir_nodata: bool = False,
    delimiter: str = "\t",
    block_rows: int = 1024,
    block_cols: int = 1024,
    decimales_coord: int = 10,
    decimales_valor: int = 6,
):
    """
    Exporta (latitud, longitud, valor) del centro de píxel para un raster MUY grande,
    procesando por ventanas y escribiendo el TXT en streaming.
    """

    os.makedirs(os.path.dirname(out_txt) or ".", exist_ok=True)

    # Si ya existe, lo reemplaza
    if os.path.exists(out_txt):
        os.remove(out_txt)

    # Encabezado
    with open(out_txt, "w", encoding="utf-8") as f:
        f.write(f"latitud{delimiter}longitud{delimiter}valor\n")

    # Evita que rasterio/GDAL intente “loggear” raro en tu consola (y provoque UnicodeDecodeError)
    os.environ.setdefault("CPL_LOG", "NUL")   # en Windows
    os.environ.setdefault("CPL_DEBUG", "OFF")

    total_escritas = 0

    with rasterio.Env():
        with rasterio.open(raster_path) as src:
            nodata = src.nodata
            crs_src = src.crs
            if crs_src is None:
                raise ValueError("El raster NO tiene CRS definido. No puedo producir lat/lon sin CRS.")

            transformer = None
            if str(crs_src).upper() != dst_epsg.upper():
                transformer = Transformer.from_crs(crs_src, dst_epsg, always_xy=True)

            height, width = src.height, src.width

            # Recorre ventanas
            for row0 in range(0, height, block_rows):
                h = min(block_rows, height - row0)
                for col0 in range(0, width, block_cols):
                    w = min(block_cols, width - col0)

                    window = Window(col0, row0, w, h)
                    data = src.read(1, window=window)

                    # Máscara de válidos
                    if incluir_nodata:
                        mask = np.ones(data.shape, dtype=bool)
                    else:
                        if nodata is not None:
                            mask = data != nodata
                        else:
                            mask = np.ones(data.shape, dtype=bool)

                    if not mask.any():
                        continue

                    # Índices locales dentro del bloque
                    rr, cc = np.where(mask)

                    # Convertir a índices globales
                    rr_g = rr + row0
                    cc_g = cc + col0

                    # Coordenadas centro (en CRS original)
                    xs, ys = xy(src.transform, rr_g, cc_g, offset="center")
                    xs = np.asarray(xs, dtype=np.float64)
                    ys = np.asarray(ys, dtype=np.float64)

                    vals = data[rr, cc].astype(np.float64)

                    # Reproyectar coordenadas a EPSG destino
                    if transformer is not None:
                        lon, lat = transformer.transform(xs, ys)
                    else:
                        lon, lat = xs, ys

                    # Redondeo
                    lat = np.round(lat, decimales_coord)
                    lon = np.round(lon, decimales_coord)
                    vals = np.round(vals, decimales_valor)

                    # Escribir incrementalmente (sin armar df gigante)
                    df = pd.DataFrame({"latitud": lat, "longitud": lon, "valor": vals})
                    df.to_csv(out_txt, sep=delimiter, index=False, header=False, mode="a", encoding="utf-8")

                    total_escritas += len(df)

    return total_escritas


def lectura_txt(ruta):
    return pd.read_csv(ruta,sep='\t',engine='python')

def Alistamiento(df):
    arreglo=['Lat_Magna','Long_Magna','altura']
    df_listo = df[arreglo]
    df_listo['Long_Magna'] = df_listo['Long_Magna'] +360
    os.makedirs('Remover',exist_ok=True)
    df_listo.to_csv('Remover/Datos_Terrestres_Matlab.txt',sep='\t', index=False, header=False)

def alistamiento_principal(ruta_datos):
    df_terrestres = lectura_txt(ruta_datos)
    arreglo=['Lat_Magna','Long_Magna','altura']
    df_terrestres = df_terrestres[arreglo]
    df_terrestres = df_terrestres.rename(columns={'longitud': 'Longitud','latitud':'Latitud'})
    Alistamiento(df_terrestres)