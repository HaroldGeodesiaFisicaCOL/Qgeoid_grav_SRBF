import gzip
import math
import shutil
import time
from pathlib import Path

import numpy as np
import rasterio
import requests
from rasterio.transform import from_origin
from requests.adapters import HTTPAdapter
from srtm import SrtmService, lat_lon_to_filename
from tqdm import tqdm
from urllib3.util.retry import Retry

# URL base del dataset público "Terrain Tiles" (AWS Open Data), formato Skadi
# No requiere autenticación ni token.
_SKADI_BASE_URL = "https://s3.amazonaws.com/elevation-tiles-prod/skadi"


def generar_modelo_srtm(
    lon_min,
    lon_max,
    lat_min,
    lat_max,
    resolucion=0.01,
    carpeta_srtm="modules/Compute_module/1_Modelos/modelo_SRTM/data",
    outputfile="modules/Compute_module/1_Modelos/modelo_SRTM/SRTM.tif",
    nodata=-99999,
    interpolado=True,
    descargar_faltantes=True,
):
    """
    Genera un GeoTIFF del modelo SRTM para una zona definida por límites
    geográficos.

    Parameters
    ----------
    descargar_faltantes : bool, optional
        Si True (por defecto), intenta descargar automáticamente los tiles
        .hgt faltantes desde el dataset público de AWS ("Terrain Tiles",
        formato Skadi) antes de fallar.
    """

    carpeta_srtm = Path(carpeta_srtm)
    outputfile = Path(outputfile)

    carpeta_srtm.mkdir(parents=True, exist_ok=True)
    outputfile.parent.mkdir(parents=True, exist_ok=True)

    _validar_tiles_srtm(
        lon_min=lon_min,
        lon_max=lon_max,
        lat_min=lat_min,
        lat_max=lat_max,
        carpeta_srtm=carpeta_srtm,
        descargar_faltantes=descargar_faltantes,
    )

    lons = np.arange(lon_min, lon_max + resolucion, resolucion)
    lats = np.arange(lat_max, lat_min - resolucion, -resolucion)

    ncols = len(lons)
    nrows = len(lats)

    print("Generando modelo SRTM...")
    print(f"Longitud: {lon_min} a {lon_max}")
    print(f"Latitud: {lat_min} a {lat_max}")
    print(f"Resolución: {resolucion} grados")
    print(f"Tamaño raster: {nrows} filas x {ncols} columnas")
    print(f"Archivo de salida: {outputfile}")

    transform = from_origin(
        west=lon_min,
        north=lat_max,
        xsize=resolucion,
        ysize=resolucion,
    )

    service = SrtmService(str(carpeta_srtm))

    perfil = {
        "driver": "GTiff",
        "height": nrows,
        "width": ncols,
        "count": 1,
        "dtype": "float32",
        "crs": "EPSG:4686",
        "transform": transform,
        "nodata": nodata,
        "compress": "deflate",
        "tiled": True,
        "blockxsize": 256,
        "blockysize": 256,
    }

    with rasterio.open(outputfile, "w", **perfil) as dst:
        for row, lat in enumerate(tqdm(lats, desc="Filas SRTM", unit="fila")):
            coords = [(float(lat), float(lon)) for lon in lons]

            if interpolado:
                valores = service.get_elevations_batch_interpolated(
                    coords,
                    default=nodata,
                )
            else:
                valores = service.get_elevations_batch(
                    coords,
                    default=int(nodata),
                )

            fila = np.asarray(valores, dtype=np.float32)
            fila[~np.isfinite(fila)] = nodata

            #Covertir los valores negativos a cero.
            fila[fila<0] = 0.0

            dst.write(fila.reshape(1, ncols), 1, window=((row, row + 1), (0, ncols)))

    print("Modelo SRTM generado correctamente.")
    return str(outputfile)


def _tile_a_carpeta_skadi(tile):
    """
    A partir de un nombre de tile tipo 'N05W073.hgt' devuelve la subcarpeta
    Skadi correspondiente, ej. 'N05'.
    """
    nombre = Path(tile).stem  # quita ".hgt" si viene incluido
    return nombre[:3]  # p.ej. 'N05' o 'S12'


def _crear_sesion_srtm():
    """
    Crea una sesión de requests con reintentos automáticos a nivel de
    conexión (útil para EOF/SSL/timeouts intermitentes en descargas largas).
    """
    sesion = requests.Session()
    reintentos = Retry(
        total=5,
        connect=5,
        read=5,
        backoff_factor=1.5,  # 0s, 1.5s, 3s, 6s, 12s...
        status_forcelist=[500, 502, 503, 504],
        allowed_methods=["GET"],
    )
    adaptador = HTTPAdapter(max_retries=reintentos, pool_maxsize=10)
    sesion.mount("https://", adaptador)
    sesion.mount("http://", adaptador)
    return sesion


def _descargar_tile_srtm(tile, carpeta_srtm, sesion=None, timeout=(10, 60), intentos=3):
    """
    Descarga un tile .hgt individual desde el dataset público Terrain Tiles
    (AWS Open Data, formato Skadi) y lo guarda descomprimido en carpeta_srtm.

    Reintenta hasta `intentos` veces ante fallos de red/SSL, además de los
    reintentos automáticos a nivel de conexión que ya trae la sesión.

    Parameters
    ----------
    sesion : requests.Session, optional
        Sesión reutilizable (recomendado para no abrir/cerrar conexión TCP
        por cada uno de los ~300 tiles). Si no se pasa, se crea una nueva.
    timeout : tuple, optional
        (timeout de conexión, timeout de lectura) en segundos.

    Returns
    -------
    bool
        True si la descarga y descompresión fueron exitosas, False si el
        tile no existe en el servidor (404, ej. zona oceánica) o si
        fallaron todos los intentos.
    """
    sesion_propia = sesion is None
    if sesion_propia:
        sesion = _crear_sesion_srtm()

    subcarpeta = _tile_a_carpeta_skadi(tile)
    url = f"{_SKADI_BASE_URL}/{subcarpeta}/{tile}.gz"

    destino_hgt = carpeta_srtm / tile
    destino_gz = carpeta_srtm / f"{tile}.gz"

    try:
        for intento in range(1, intentos + 1):
            try:
                with sesion.get(url, stream=True, timeout=timeout) as resp:
                    if resp.status_code == 404:
                        # Tile inexistente en el servidor (frecuente en
                        # zonas oceánicas o sin cobertura). No reintentar.
                        return False
                    resp.raise_for_status()

                    with open(destino_gz, "wb") as f:
                        # iter_content (no resp.raw) para que requests
                        # traduzca errores de urllib3/ssl a excepciones
                        # manejables en vez de dejarlos pasar "crudos".
                        for chunk in resp.iter_content(chunk_size=1024 * 256):
                            if chunk:
                                f.write(chunk)

                with gzip.open(destino_gz, "rb") as f_in, open(destino_hgt, "wb") as f_out:
                    shutil.copyfileobj(f_in, f_out)

                return True

            except (requests.RequestException, OSError) as exc:
                destino_gz.unlink(missing_ok=True)
                if intento == intentos:
                    print(f"  ! Error descargando {tile} tras {intentos} intento(s): {exc}")
                    return False
                time.sleep(2 * intento)

        return False

    finally:
        destino_gz.unlink(missing_ok=True)
        if sesion_propia:
            sesion.close()



def _validar_tiles_srtm(
    lon_min, lon_max, lat_min, lat_max, carpeta_srtm, descargar_faltantes=True
):
    """
    Verifica qué archivos SRTM .hgt o .hgt.zip son necesarios para la zona.
    Si descargar_faltantes=True, intenta descargarlos automáticamente antes
    de fallar.
    """

    lat_inicio = math.floor(lat_min)
    lat_fin = math.ceil(lat_max)
    lon_inicio = math.floor(lon_min)
    lon_fin = math.ceil(lon_max)

    tiles_necesarios = []
    for lat in range(lat_inicio, lat_fin):
        for lon in range(lon_inicio, lon_fin):
            tiles_necesarios.append(lat_lon_to_filename(lat, lon))

    tiles_faltantes = [
        tile
        for tile in tiles_necesarios
        if not (carpeta_srtm / tile).exists()
        and not (carpeta_srtm / f"{tile}.zip").exists()
    ]

    if tiles_faltantes and descargar_faltantes:
        print(f"\nDescargando {len(tiles_faltantes)} tile(s) SRTM faltante(s)...")
        aun_faltantes = []
        sesion = _crear_sesion_srtm()
        try:
            for tile in tqdm(tiles_faltantes, desc="Descargando tiles", unit="tile"):
                if not _descargar_tile_srtm(tile, carpeta_srtm, sesion=sesion):
                    aun_faltantes.append(tile)
        finally:
            sesion.close()
        tiles_faltantes = aun_faltantes

    if tiles_faltantes:
        print("\nFaltan archivos SRTM para la zona solicitada:")
        for tile in tiles_faltantes:
            print(f"  - {tile} o {tile}.zip")

        raise FileNotFoundError(
            "\nNo se pudieron obtener todos los tiles .hgt/.hgt.zip necesarios "
            "(puede que la descarga automática haya fallado o que el tile no "
            "exista en el servidor, ej. zona oceánica).\n"
            f"Guarda manualmente los archivos faltantes en:\n{carpeta_srtm.resolve()}"
        )
