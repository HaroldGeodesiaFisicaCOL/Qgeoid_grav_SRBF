import gdown
import os

def dowloand_Earth_surface(link_carpeta: str, carpeta_destino: str):
    """
    Descarga todo el contenido de una carpeta de Google Drive.

    Args:
        link_carpeta (str): URL de la carpeta compartida de Drive
        carpeta_destino (str): Carpeta local donde se guardará el contenido

    Returns:
        list: Rutas de los archivos descargados
    """
    os.makedirs(carpeta_destino, exist_ok=True)

    print(f"Descargando carpeta: {link_carpeta}")
    archivos = gdown.download_folder(
        url=link_carpeta,
        output=carpeta_destino,
        quiet=False,
        use_cookies=False
    )

    print(f"Descarga completa en: {carpeta_destino}")
    return archivos
