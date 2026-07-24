import re
import time
from pathlib import Path

import requests
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from tqdm import tqdm


def descargar_modelo_icgem(
    modelo,
    grado=None,
    ruta=None,
    tipo="global",
    sobrescribir=False,
    timeout=120,
    max_reintentos=8,
):
    """
    Descarga un modelo del ICGEM.

    Parameters
    ----------
    modelo : str
        Nombre del modelo.
        Ejemplos:
            "XGM2019"
            "EIGEN-6C4"
            "Earth2014"
            "dV_ELL_Earth2014_plusGRS80"

    grado : int, optional
        Grado máximo del modelo.
        Ejemplos:
            760
            2159
            5540

        Si es None se descarga el primer modelo encontrado.

    ruta : str o Path, optional
        Carpeta donde guardar el archivo.
        Si es None se crea "Modelos_ICGEM" en el directorio actual.

    tipo : {"global","topografia"}

    sobrescribir : bool

    timeout : int

    max_reintentos : int
        Número de intentos ante cortes de conexión durante la
        descarga (archivos grandes son más propensos a esto).
        La descarga se reanuda desde donde se cortó, no repite
        desde cero.

    Returns
    -------
    str
        Ruta del archivo descargado.
    """

    modelo = modelo.replace(".gfc", "").strip()

    # -----------------------------------------------------
    # Carpeta destino
    # -----------------------------------------------------

    if ruta is None:
        carpeta = Path("Modelos_ICGEM")
    else:
        carpeta = Path(ruta)

    carpeta.mkdir(parents=True, exist_ok=True)

    # -----------------------------------------------------
    # Catálogo
    # -----------------------------------------------------

    if tipo.lower() == "global":
        catalogo = "https://icgem.gfz.de/tom_longtime"

    elif tipo.lower() in ("topografia", "topographic", "topo"):
        catalogo = "https://icgem.gfz.de/tom_reltopo"

    else:
        raise ValueError("tipo debe ser 'global' o 'topografia'")

    # -----------------------------------------------------
    # Sesión HTTP
    # -----------------------------------------------------

    session = requests.Session()

    retry = Retry(
        total=5,
        backoff_factor=2,
        status_forcelist=[500, 502, 503, 504],
    )

    session.mount(
        "https://",
        HTTPAdapter(max_retries=retry)
    )

    print("Consultando catálogo...")

    html = session.get(catalogo, timeout=timeout)

    if html.status_code != 200:
        raise RuntimeError("No fue posible acceder al catálogo.")

    soup = BeautifulSoup(html.text, "html.parser")

    # -----------------------------------------------------
    # Buscar enlace
    # -----------------------------------------------------

    enlace = None
    nombre_archivo = None

    candidatos = []  # (nombre, nombre_lower, href)

    for a in soup.find_all("a", href=True):

        href = a["href"]

        if not href.lower().endswith(".gfc"):
            continue

        nombre = Path(href).name
        nombre_lower = nombre.lower()

        if modelo.lower() not in nombre_lower:
            continue

        candidatos.append((nombre, nombre_lower, href))

    if not candidatos:
        raise RuntimeError(
            f"No se encontró el modelo '{modelo}' "
            f"(grado={grado})."
        )

    # -------------------------------------------------------------
    # 1) Prioridad máxima: coincidencia EXACTA del nombre de archivo.
    #    Cubre tanto el caso grado=None como el caso en que ya se
    #    pasó el nombre completo del modelo (típico en topografía,
    #    donde el grado no siempre es un sufijo separado del nombre).
    # -------------------------------------------------------------

    exacto = f"{modelo.lower()}.gfc"

    exacta = [c for c in candidatos if c[1] == exacto]

    if exacta:
        nombre_archivo, _, enlace = exacta[0]

    elif grado is None:

        # Sin coincidencia exacta y sin grado: tomar el nombre
        # más corto (el que tenga menos sufijos añadidos).
        nombre_archivo, _, enlace = min(
            candidatos, key=lambda c: len(c[1])
        )

    else:

        # 2) Sin coincidencia exacta pero con grado: aplicar la
        #    lógica de sufijo numérico (pensada para modelos
        #    globales tipo XGM/EIGEN).
        grado = int(grado)

        for nombre, nombre_lower, href in candidatos:

            if grado == 760:

                if re.fullmatch(
                    rf"{re.escape(modelo.lower())}\.gfc",
                    nombre_lower,
                ):
                    enlace, nombre_archivo = href, nombre
                    break

            else:

                patron = rf"{re.escape(modelo.lower())}.*[_-]{grado}\.gfc"

                if re.search(patron, nombre_lower):
                    enlace, nombre_archivo = href, nombre
                    break

                # modelos como XGM2019e.gfc (5540)

                if (
                    grado == 5540
                    and nombre_lower == f"{modelo.lower()}e.gfc"
                ):
                    enlace, nombre_archivo = href, nombre
                    break

        if enlace is None:
            raise RuntimeError(
                f"No se encontró el modelo '{modelo}' "
                f"(grado={grado})."
            )

    if enlace.startswith("/"):
        enlace = "https://icgem.gfz.de" + enlace

    elif not enlace.startswith("http"):
        enlace = "https://icgem.gfz.de/" + enlace

    archivo = carpeta / nombre_archivo

    if archivo.exists() and not sobrescribir:

        print(f"\n✓ El archivo ya existe:\n{archivo}")

        return str(archivo)

    print("\nModelo :", modelo)

    if grado is not None:
        print("Grado  :", grado)

    print("Archivo:", nombre_archivo)
    print("URL    :", enlace)

    # -----------------------------------------------------
    # Descargar (con reintentos y reanudación)
    # -----------------------------------------------------

    inicio = time.time()

    archivo_temp = archivo.with_suffix(archivo.suffix + ".part")

    total = None

    for intento in range(1, max_reintentos + 1):

        descargado = archivo_temp.stat().st_size if archivo_temp.exists() else 0

        headers = {}

        if descargado > 0:
            headers["Range"] = f"bytes={descargado}-"

        try:

            respuesta = session.get(
                enlace,
                stream=True,
                timeout=timeout,
                headers=headers,
            )

            if respuesta.status_code == 416:
                # El servidor dice que ya no queda nada por leer:
                # asumimos que el archivo ya está completo.
                break

            if respuesta.status_code not in (200, 206):
                raise RuntimeError(
                    f"Error HTTP {respuesta.status_code}"
                )

            # Si pedimos Range pero el servidor respondió 200
            # (no soporta reanudación), reiniciamos desde cero.
            if descargado > 0 and respuesta.status_code == 200:
                descargado = 0

            modo = "ab" if descargado > 0 else "wb"

            if total is None:
                if respuesta.status_code == 206:
                    content_range = respuesta.headers.get(
                        "content-range", ""
                    )
                    if "/" in content_range:
                        total = int(content_range.split("/")[-1])
                else:
                    total = int(
                        respuesta.headers.get("content-length", 0)
                    )

            with open(archivo_temp, modo) as f:

                barra = tqdm(
                    total=total,
                    initial=descargado,
                    unit="B",
                    unit_scale=True,
                    unit_divisor=1024,
                    desc="Descargando",
                )

                for chunk in respuesta.iter_content(1024 * 1024):

                    if chunk:
                        f.write(chunk)
                        barra.update(len(chunk))

                barra.close()

            if total and archivo_temp.stat().st_size < total:
                raise requests.exceptions.ChunkedEncodingError(
                    "Descarga incompleta, se reintentará."
                )

            break  # descarga completa

        except (
            requests.exceptions.ChunkedEncodingError,
            requests.exceptions.ConnectionError,
            requests.exceptions.Timeout,
        ) as e:

            print(
                f"\n⚠ Conexión interrumpida (intento {intento}/{max_reintentos}): {e}"
            )

            if intento == max_reintentos:
                raise RuntimeError(
                    "No fue posible completar la descarga tras "
                    f"{max_reintentos} intentos."
                ) from e

            espera = min(2 ** intento, 60)

            print(f"Reintentando en {espera} s...")

            time.sleep(espera)

    archivo_temp.rename(archivo)

    # -----------------------------------------------------
    # Verificación
    # -----------------------------------------------------

    with open(archivo, "rb") as f:

        cabecera = f.read(200).lower()

    if b"<html" in cabecera:

        archivo.unlink()

        raise RuntimeError(
            "La descarga devolvió una página HTML."
        )

    tiempo = time.time() - inicio

    tam = archivo.stat().st_size / 1024 / 1024

    print("\n✓ Descarga terminada")
    print(f"Archivo   : {archivo}")
    print(f"Tamaño    : {tam:.2f} MB")
    print(f"Tiempo    : {tiempo:.1f} s")
    print(f"Velocidad : {tam/tiempo:.2f} MB/s")

    return str(archivo)
