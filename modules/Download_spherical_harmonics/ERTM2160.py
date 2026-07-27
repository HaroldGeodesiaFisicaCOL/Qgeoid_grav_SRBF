"""
ertm2160_2013_v2.py
--------------------------------------------------------------------------
Traducción a Python de la función MATLAB "ertm2160_2013_v2.m" (Christian
Hirt, WA Centre for Geodesy, Curtin University), incluyendo la exportación
a GeoTIFF que originalmente estaba en los scripts de ejecución
"Ejecucion_Anomalias_Altura.m" y "Ejecucion_Perturbaciones_grav.m".

Se expone UNA sola función pública, `generar_modelo_ertm2160`, que:

  1. Localiza y lee los tiles binarios de ERTM2160 necesarios para el área
     objetivo.
  2. Ensambla la grilla de salida (con o sin interpolación, según facX/facY).
  3. Aplica el factor de escala del funcional elegido.
  4. Opcionalmente exporta el resultado como GeoTIFF (EPSG configurable,
     por defecto 4686 = MAGNA-SIRGAS, igual que en los scripts originales).

Para usarla en un main.py, basta con llamarla dos veces: una con
functional='geoid' (anomalías de altura) y otra con functional='gravity'
(perturbaciones de gravedad).

Dependencias:  numpy, scipy, rasterio
    pip install numpy scipy rasterio
--------------------------------------------------------------------------
"""

import os
import numpy as np
from scipy.interpolate import RegularGridInterpolator

try:
    import rasterio
    # --------------------------------------------------------------
    # Fix: en Windows es común tener otra instalación de PROJ (p.ej.
    # la de PostGIS/PostgreSQL) registrada en las variables de entorno
    # PROJ_LIB / PROJ_DATA. Eso hace que rasterio intente usar un
    # proj.db incompatible y falle con CRSError al resolver el EPSG.
    # Forzamos aquí el proj.db que trae el propio paquete rasterio.
    # --------------------------------------------------------------
    _proj_dirs_candidatas = [
        os.path.join(os.path.dirname(rasterio.__file__), 'proj_data'),
        os.path.join(os.path.dirname(rasterio.__file__), 'gdal_data', '..', 'proj_data'),
    ]
    for _d in _proj_dirs_candidatas:
        _d = os.path.normpath(_d)
        if os.path.isdir(_d):
            os.environ['PROJ_LIB'] = _d
            os.environ['PROJ_DATA'] = _d
            break

    from rasterio.transform import from_bounds
    _RASTERIO_OK = True
except ImportError:
    _RASTERIO_OK = False


# --------------------------------------------------------------------------
# Función auxiliar: nombre de archivo del tile (equivalente a getfilename2012)
# --------------------------------------------------------------------------
def _nombre_archivo_tile(lat, lon):
    """
    Reproduce getfilename2012.m: construye el nombre base del tile
    (sin sufijo) a partir de la esquina SW del tile, p.ej. 'N05W075'.
    """
    d_int = int(lat)
    c_int = int(lon)

    prefijo_lat = 'N' if d_int >= 0 else 'S'
    if abs(d_int) < 10:
        prefijo_lat += '0'
    prefijo_lat += f'{abs(d_int)}'

    prefijo_lon = 'E' if c_int >= 0 else 'W'
    if abs(c_int) < 100:
        prefijo_lon += '0'
    if abs(c_int) < 10:
        prefijo_lon += '0'
    prefijo_lon += f'{abs(c_int)}'

    return prefijo_lat + prefijo_lon


# --------------------------------------------------------------------------
# Función principal
# --------------------------------------------------------------------------
def generar_modelo_ertm2160(functional, minlon1, maxlon1, minlat1, maxlat1,
                             facX, facY, path_basis, nodataval, method,
                             outputfile=None, epsg=4686):
    """
    Extrae un funcional del modelo ERTM2160 para un área objetivo y,
    opcionalmente, lo exporta como GeoTIFF.

    Parámetros
    ----------
    functional : str
        'geoid'   -> anomalías de altura / ondulación del geoide (m)
        'gravity' -> perturbaciones de gravedad (mGal)
        'xi', 'eta', 'dem' también soportados, igual que en el .m original.
    minlon1, maxlon1, minlat1, maxlat1 : float
        Límites geodésicos del área objetivo (grados).
    facX, facY : float
        Factores de escala de resolución. facX=facY=1 => sin interpolación.
    path_basis : str
        Ruta base donde están las carpetas de datos ERTM2160
        (p.ej. contiene subcarpetas 'geoid', 'dg', 'xi', 'eta', 'dem').
    nodataval : float
        Valor asignado donde no hay datos (usar np.nan si se desea igual
        que en el script original).
    method : str
        'nearest', 'linear' o 'cubic' (equivalente a 'spline' del .m).
        Se ignora si facX=facY=1.
    outputfile : str, opcional
        Ruta del GeoTIFF de salida. Si es None, no se exporta archivo.
    epsg : int
        Código EPSG de referencia (4686 = MAGNA-SIRGAS, por defecto).

    Retorna
    -------
    X, Y, Z : np.ndarray
        Matrices tipo meshgrid con longitudes, latitudes y el funcional
        extraído (o interpolado).
    """

    # ------------------------------------------------------------------
    # 1) Parámetros constantes según el funcional
    # ------------------------------------------------------------------
    tilesize = 5
    resolution = 7.2 / 3600.0
    tileelems = 2500
    dtype_bin = '>i2'          # int16 big-endian (ieee-be)
    init_nodata = -2 ** 15

    if functional == 'geoid':
        conv_factor = 1000.0   # mm -> m
        suffix = '.ha'
        path_tiles = os.path.join(path_basis, 'geoid')
    elif functional == 'gravity':
        conv_factor = 10.0     # 0.1 mGal -> mGal
        suffix = '.dg'
        path_tiles = os.path.join(path_basis, 'dg')
    elif functional == 'xi':
        conv_factor = 10.0     # 0.1 arc-sec -> arc-sec
        suffix = '.xi'
        path_tiles = os.path.join(path_basis, 'xi')
    elif functional == 'eta':
        conv_factor = 10.0
        suffix = '.eta'
        path_tiles = os.path.join(path_basis, 'eta')
    elif functional == 'dem':
        conv_factor = 1.0
        suffix = '.dem'
        path_tiles = os.path.join(path_basis, 'dem')
    else:
        raise ValueError(f"functional desconocido: {functional!r}")

    if facX == 1 and facY == 1:
        interpflag = False
        tolarea = 0.0
    else:
        interpflag = True
        tolarea = 0.1

    # ------------------------------------------------------------------
    # 2) Extender el área de selección para calzar con los tiles
    # ------------------------------------------------------------------
    minlon2 = minlon1 - tolarea
    maxlon2 = maxlon1 + tolarea
    minlat2 = minlat1 - tolarea
    maxlat2 = maxlat1 + tolarea

    Sall_minlon = np.floor(minlon2 / tilesize) * tilesize
    Sall_maxlon = np.floor(maxlon2 / tilesize) * tilesize
    Sall_minlat = np.floor(minlat2 / tilesize) * tilesize
    Sall_maxlat = np.floor(maxlat2 / tilesize) * tilesize

    lonvec = np.arange(Sall_minlon, Sall_maxlon + tilesize, tilesize)
    latvec = np.arange(Sall_minlat, Sall_maxlat + tilesize, tilesize)

    tiles = [(lo, la) for la in latvec for lo in lonvec]

    # ------------------------------------------------------------------
    # 3) Definir la grilla destino usando aritmética entera
    #    (0.01 arc-sec, igual que en el .m, para evitar errores de
    #    redondeo en coma flotante)
    # ------------------------------------------------------------------
    fac = 3600 * 100
    r = int(round(resolution * fac))
    r2 = r // 2

    Amin_lat = int(np.floor(minlat2 * fac / r) * r + r2)
    Amax_lat = int(np.floor(maxlat2 * fac / r) * r - r2)
    Amin_lon = int(np.floor(minlon2 * fac / r) * r + r2)
    Amax_lon = int(np.floor(maxlon2 * fac / r) * r - r2)

    Avec_lat = np.arange(Amin_lat, Amax_lat + r, r)
    Avec_lon = np.arange(Amin_lon, Amax_lon + r, r)

    An1 = len(Avec_lat)
    Am1 = len(Avec_lon)

    A = np.full((An1, Am1), np.nan, dtype=np.float64)

    # ------------------------------------------------------------------
    # 4) Rellenar la matriz con los datos de cada tile binario
    # ------------------------------------------------------------------
    for (slonmin, slatmin) in tiles:
        slatmax = slatmin + tilesize
        slonmax = slonmin + tilesize

        nombre_base = _nombre_archivo_tile(slatmin, slonmin)
        nombre_archivo = nombre_base + suffix
        ruta_archivo = os.path.join(path_tiles, nombre_archivo)

        Smin_lat = int(round(slatmin * fac + r2))
        Smax_lat = int(round(slatmax * fac - r2))
        Smin_lon = int(round(slonmin * fac + r2))
        Smax_lon = int(round(slonmax * fac - r2))

        Svec_lat = np.arange(Smin_lat, Smax_lat + r, r)
        Svec_lon = np.arange(Smin_lon, Smax_lon + r, r)

        a_lat = np.intersect1d(Svec_lat, Avec_lat)
        a_lon = np.intersect1d(Svec_lon, Avec_lon)

        if len(a_lat) == 0 or len(a_lon) == 0:
            print(' este tile no intersecta con el área objetivo')
            continue

        Aix_lat_min = int(np.searchsorted(Avec_lat, a_lat[0]))
        Aix_lat_max = int(np.searchsorted(Avec_lat, a_lat[-1]))
        Six_lat_min = int(np.searchsorted(Svec_lat, a_lat[0]))
        Six_lat_max = int(np.searchsorted(Svec_lat, a_lat[-1]))

        Aix_lon_min = int(np.searchsorted(Avec_lon, a_lon[0]))
        Aix_lon_max = int(np.searchsorted(Avec_lon, a_lon[-1]))
        Six_lon_min = int(np.searchsorted(Svec_lon, a_lon[0]))
        Six_lon_max = int(np.searchsorted(Svec_lon, a_lon[-1]))

        if not os.path.exists(ruta_archivo):
            print(f' el archivo {nombre_archivo} no existe')
            continue

        print(' leer archivo')
        print(' ' + ruta_archivo)
        with open(ruta_archivo, 'rb') as f:
            S = np.fromfile(f, dtype=dtype_bin, count=tileelems * tileelems)
        print(' lectura completa')
        # OJO: MATLAB reshape(S,te,te) rellena por COLUMNAS (Fortran).
        # Hay que replicar ese orden aquí; si se usa el orden por defecto
        # de NumPy (C / por filas), el contenido de cada tile queda
        # transpuesto/revuelto (ruido tipo "speckle" en el resultado).
        S = S.reshape((tileelems, tileelems), order='F').astype(np.float64)
        S[S < init_nodata + 1] = np.nan

        A[Aix_lat_min:Aix_lat_max + 1, Aix_lon_min:Aix_lon_max + 1] = \
            S[Six_lat_min:Six_lat_max + 1, Six_lon_min:Six_lon_max + 1]

    print(f' puntos sin datos (NaN): {int(np.isnan(A).sum())}')

    # ------------------------------------------------------------------
    # 5) Construir las matrices X, Y (meshgrid) en grados
    # ------------------------------------------------------------------
    Avec_lon_deg = Avec_lon.astype(np.float64) / fac
    Avec_lat_deg = Avec_lat.astype(np.float64) / fac

    AX, AY = np.meshgrid(Avec_lon_deg, Avec_lat_deg)

    # ------------------------------------------------------------------
    # 6) Interpolar solo si facX/facY != 1
    # ------------------------------------------------------------------
    metodo_map = {'nearest': 'nearest', 'linear': 'linear',
                  'cubic': 'cubic', 'spline': 'cubic'}
    metodo_scipy = metodo_map.get(method, 'cubic')

    if interpflag:
        print(f' interpolar {method}')

        res_targetX = resolution * facX
        res_targetY = resolution * facY

        B_veclon = np.arange(minlon1, maxlon1 + res_targetX / 2, res_targetX)
        B_veclat = np.arange(minlat1, maxlat1 + res_targetY / 2, res_targetY)

        BX, BY = np.meshgrid(B_veclon, B_veclat)

        interpolador = RegularGridInterpolator(
            (Avec_lat_deg, Avec_lon_deg), A,
            method=metodo_scipy, bounds_error=False, fill_value=np.nan)

        puntos = np.column_stack([BY.ravel(), BX.ravel()])
        Z = interpolador(puntos).reshape(BY.shape)

        X, Y = BX, BY
        print(' interpolación finalizada')
    else:
        print(' se pasan los datos extraídos directamente.')
        X, Y, Z = AX, AY, A

    # ------------------------------------------------------------------
    # 7) Aplicar el factor de escala del funcional
    # ------------------------------------------------------------------
    Z = Z / conv_factor

    # ------------------------------------------------------------------
    # 8) Asignar el nodataval definido por el usuario
    # ------------------------------------------------------------------
    Z = np.where(np.isnan(Z), nodataval, Z)
    print(np.nanmin(Z))
    print(np.nanmax(Z))
    # ------------------------------------------------------------------
    # 9) Exportar a GeoTIFF (opcional)
    # ------------------------------------------------------------------
    if outputfile is not None:
        if not _RASTERIO_OK:
            raise ImportError(
                "rasterio no está instalado. Instálalo con "
                "'pip install rasterio' para poder exportar el GeoTIFF."
            )

        Z_out = Z
        # Si Y crece de sur a norte (fila 0 = sur), se voltea para que la
        # fila 0 del GeoTIFF corresponda al norte, como exige el formato.
        if Y[0, 0] < Y[-1, 0]:
            Z_out = np.flipud(Z_out)

        west, east = float(np.min(X)), float(np.max(X))
        south, north = float(np.min(Y)), float(np.max(Y))
        n_rows, n_cols = Z_out.shape

        transform = from_bounds(west, south, east, north, n_cols, n_rows)

        os.makedirs(os.path.dirname(outputfile) or '.', exist_ok=True)
        with rasterio.open(
            outputfile, 'w',
            driver='GTiff',
            height=n_rows,
            width=n_cols,
            count=1,
            dtype=Z_out.dtype,
            crs=f'EPSG:{epsg}',
            transform=transform,
        ) as dst:
            dst.write(Z_out, 1)

        print(f'Archivo GeoTIFF exportado con EPSG:{epsg} con éxito: {outputfile}')



# if __name__ == '__main__':
#     # Ejemplo mínimo de uso (ajusta rutas y límites a tu caso real)
#     path_ertm2160 = 'D:/2026/Modelos_Usados/ERTM2160/data'

#     # Anomalías de altura (geoide)
#     X1, Y1, Z1 = generar_modelo_ertm2160(
#         'geoid', -80.02083333, -64.97916667, -5.01250000, 13.21250000,
#         1.0, 1.0, path_ertm2160, np.nan, 'cubic',
#         outputfile='D:/2026/Modelos_Usados/ERTM2160/Anomalias_Altura_ERTM_2160.tif'
#     )

#     # Perturbaciones de gravedad
#     X2, Y2, Z2 = generar_modelo_ertm2160(
#         'gravity', -79.30000000, -66.63840000, -4.77630000, 12.88330000,
#         1.0, 1.0, path_ertm2160, np.nan, 'cubic',
#         outputfile='D:/2026/Modelos_Usados/ERTM2160/Perturbaciones_gravedad_ERTM_2160.tif'
#     )
