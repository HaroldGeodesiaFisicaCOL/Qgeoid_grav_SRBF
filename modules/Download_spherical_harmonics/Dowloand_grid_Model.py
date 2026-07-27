import os
from math import floor
from urllib.parse import urljoin
import re

import requests
from bs4 import BeautifulSoup
from tqdm import tqdm


def deg_to_str(deg, is_lat=True):
    """Convierte una coordenada a formato SRTM (ej. N04, W075)"""
    hemi = 'N' if is_lat and deg >= 0 else 'S' if is_lat else 'E' if deg >= 0 else 'W'
    deg_abs = abs(floor(deg))
    return f"{hemi}{deg_abs:02d}" if is_lat else f"{hemi}{deg_abs:03d}"

def tile_folder(lat, lon):
    """Devuelve la carpeta de 30x30° (ej. N00W090)"""
    lat_base = 30 * (floor(lat / 30))
    lon_base = 30 * (floor(lon / 30))
    return f"{deg_to_str(lat_base)}{deg_to_str(lon_base, is_lat=False)}"

def generate_download_list(lat_min, lat_max, lon_min, lon_max):
    files = []
    for lat in range(floor(lat_min), floor(lat_max) + 1):
        for lon in range(floor(lon_min), floor(lon_max) + 1):
            lat_str = deg_to_str(lat)
            lon_str = deg_to_str(lon, is_lat=False)
            tile = f"{lat_str}{lon_str}_res.bin"
            folder = tile_folder(lat, lon)
            files.append((folder, tile))
    return files

def download_tiles(files, base_url, base_local_path):
    # Usamos tqdm para la barra de progreso
    for folder, filename in tqdm(files, desc="Descargando archivos", unit="archivo", ncols=100):
        url = f"{base_url}/{folder}/{filename}"
        local_dir = os.path.join(base_local_path, folder)
        os.makedirs(local_dir, exist_ok=True)
        local_path = os.path.join(local_dir, filename)

        if os.path.exists(local_path):
            print(f" Ya existe: {filename}")
            continue

        print(f" Descargando: {filename} desde {url}")
        response = requests.get(url, stream=True)
        if response.status_code == 200:
            with open(local_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=1024):
                    f.write(chunk)
            print(f" Guardado en: {local_path}")
        else:
            print(f" No encontrado: {filename} (Código: {response.status_code})")

def dowloand_all_folder(
    url="https://ddfe.curtin.edu.au/gravitymodels/ERTM2160/data",
    carpeta_salida="data",
    sobrescribir=False,
    timeout=120,
):
    """
    Descarga todos los archivos contenidos en la carpeta geoid
    del servidor ERTM2160.

    Parameters
    ----------
    url : str
        URL del directorio.
    carpeta_salida : str
        Carpeta donde se guardarán los archivos.
    sobrescribir : bool
        Si True, vuelve a descargar archivos existentes.
    timeout : int
        Tiempo máximo de espera (s).
    """

    os.makedirs(carpeta_salida, exist_ok=True)

    print("Leyendo índice del servidor...")

    session = requests.Session()

    r = session.get(url, timeout=timeout)
    r.raise_for_status()

    soup = BeautifulSoup(r.text, "html.parser")

    enlaces = []

    for a in soup.find_all("a"):
        href = a.get("href")

        if not isinstance(href, str):
            continue

        if href == "../" or href.startswith("?"):
            continue

        if href.endswith("/"):
            continue

        enlaces.append(href)

    print(f"Se encontraron {len(enlaces)} archivos.\n")

    for archivo in tqdm(enlaces, unit="archivo"):

        destino = os.path.join(carpeta_salida, archivo)

        if os.path.exists(destino) and not sobrescribir:
            continue

        archivo_url = urljoin(url, archivo)

        try:
            with session.get(archivo_url, stream=True, timeout=timeout) as r:

                if r.status_code != 200:
                    print(f"No se pudo descargar {archivo}")
                    continue

                total = int(r.headers.get("Content-Length", 0))

                with open(destino, "wb") as f:

                    if total > 0:
                        barra = tqdm(
                            total=total,
                            unit="B",
                            unit_scale=True,
                            leave=False,
                            desc=archivo,
                        )

                        for chunk in r.iter_content(chunk_size=1024 * 1024):
                            if chunk:
                                f.write(chunk)
                                barra.update(len(chunk))

                        barra.close()

                    else:
                        for chunk in r.iter_content(chunk_size=1024 * 1024):
                            if chunk:
                                f.write(chunk)

        except Exception as e:
            print(f"Error descargando {archivo}: {e}")

    print("\nDescarga finalizada.")


def generate_download_ERTM2160(
    lat_min,
    lat_max,
    lon_min,
    lon_max,
    base_url="https://ddfe.curtin.edu.au/gravitymodels/ERTM2160/data/geoid",
    output_dir="ERTM2160",
    timeout=120,
):
    """
    Descarga automáticamente los mosaicos ERTM2160 que intersectan el
    rectángulo definido por lat_min, lat_max, lon_min y lon_max.

    No supone ninguna extensión (.ha, .bin, .zip, ...). Obtiene el listado
    directamente del servidor y descarga los archivos encontrados.
    """

    session = requests.Session()

    print("Leyendo índice del servidor...")
    r = session.get(base_url, timeout=timeout)
    r.raise_for_status()

    soup = BeautifulSoup(r.text, "html.parser")

    archivos = []

    for a in soup.find_all("a"):

        href = a.get("href")

        if href is None:
            continue

        # ignorar directorios e imágenes
        if href.endswith("/"):
            continue

        if href.lower().endswith(".png"):
            continue

        m = re.search(r'([NS]\d{2})([EW]\d{3})', href)

        if m is None:
            continue

        lat0 = int(m.group(1)[1:])
        if m.group(1)[0] == "S":
            lat0 = -lat0

        lon0 = int(m.group(2)[1:])
        if m.group(2)[0] == "W":
            lon0 = -lon0

        # mosaico de 5x5°
        if (
            lat0 + 5 > lat_min
            and lat0 < lat_max
            and lon0 + 5 > lon_min
            and lon0 < lon_max
        ):
            archivos.append(href)

    if not archivos:
        print("No se encontraron archivos para esa región.")
        return

    os.makedirs(output_dir, exist_ok=True)

    print(f"\nSe encontraron {len(archivos)} archivos.\n")

    for archivo in tqdm(archivos):

        url = f"{base_url}/{archivo}"
        destino = os.path.join(output_dir, archivo)

        if os.path.exists(destino):
            continue

        r = session.get(url, stream=True, timeout=timeout)

        if r.status_code != 200:
            print(f"No se pudo descargar {archivo}")
            continue

        with open(destino, "wb") as f:
            for chunk in r.iter_content(1024 * 1024):
                if chunk:
                    f.write(chunk)

    print("Descarga terminada.")
