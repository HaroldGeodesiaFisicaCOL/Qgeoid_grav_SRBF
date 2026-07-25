# -*- coding: utf-8 -*-
"""
Created on Tue Mar 18 17:29:51 2025

@author: harold Olarte
"""

import tempfile
import numpy as np
import pygmt
import rasterio
import xarray as xr


# ==========================
# Helpers (SOLO para QGEOIDE)
# ==========================
def _tiff_to_xarray_nomask(path):
    """
    Lee la banda 1 del GeoTIFF ignorando mask/alpha/nodata, y devuelve
    un xarray.DataArray con dims ("lat","lon") y coords lon/lat.
    Asume North-Up (b=d~0 en Affine). Invierte filas si la latitud decrece.
    """
    with rasterio.open(path) as src:
        z = src.read(1, masked=False).astype(float)
        T = src.transform
        ncols, nrows = src.width, src.height

        # Centros de píxel
        xs = np.arange(ncols) + 0.5
        ys = np.arange(nrows) + 0.5
        lons = T.c + xs * T.a
        lats = T.f + ys * T.e  # suele ser decreciente (T.e < 0)

        if lats[1] < lats[0]:
            lats = lats[::-1]
            z = z[::-1, :]

        return xr.DataArray(z, coords={"lat": lats, "lon": lons}, dims=("lat", "lon"), name="band1")


def _zrange_from_da(da):
    """Min y max del DataArray, ignorando no finitos (evita CPT degenerado)."""
    z = np.asarray(da.values, dtype=float)
    z = z[np.isfinite(z)]
    if z.size == 0:
        return -1e-6, 1e-6
    zmin, zmax = float(z.min()), float(z.max())
    if zmin == zmax:
        zmin -= 1e-6
        zmax += 1e-6
    return zmin, zmax


def _make_cpt_from_range(zmin, zmax, cmap="turbo", nsteps=200):
    """Crea un CPT temporal continuo con rango [zmin, zmax] y devuelve su ruta."""
    dz = max((zmax - zmin) / nsteps, 1e-9)
    cpt_file = tempfile.NamedTemporaryFile(delete=False, suffix=".cpt").name
    pygmt.makecpt(cmap=cmap, series=[zmin, zmax, dz], continuous=True, output=cpt_file)
    return cpt_file


# ==========================
# QGEOIDE (AJUSTADO SOLO AQUÍ)
# ==========================
def mapa_nacional(ruta_qgeoide_tiff, ruta_mapa_pdf, grid):
    fig = pygmt.Figure()
    region = [-79, -66.8, -4.1, 13]

    # Fondo de relieve
    fig.grdimage(grid=grid, region=region, projection="M15c", frame="a", cmap="geo")

    # Qgeoide: sin máscara + CPT min/max del raster
    da = _tiff_to_xarray_nomask(ruta_qgeoide_tiff)
    zmin, zmax = _zrange_from_da(da)
    cpt = _make_cpt_from_range(zmin, zmax, cmap="jet")

    fig.grdimage(
        grid=da,
        region=region,
        projection="M15c",
        frame="af",
        cmap=cpt
    )

    fig.colorbar(cmap=cpt, frame=["a2", "x+lAnomalía de Altura", "y+lm"])

    # Etiquetas países (como tenías)
    countries = [
        {"lon": -71, "lat": 4, "name": "COLOMBIA"},
        {"lon": -68, "lat": 9.0, "name": "VENEZUELA"},
        {"lon": -68, "lat": -2, "name": "BRASIL"},
        {"lon": -75, "lat": -2, "name": "PERÚ"},
        {"lon": -78, "lat": -1, "name": "ECUADOR"},
    ]
    for c in countries:
        fig.text(x=c["lon"], y=c["lat"], text=c["name"],
                 font="12p,Times-Roman,black", justify="CM")

    fig.coast(projection="M15c", borders="1/0.5p", shorelines="1/0.5p", frame="ag")
    fig.savefig(ruta_mapa_pdf)
    #fig.show()


def mapa_local(ruta_qgeoide_tiff, ruta_mapa_pdf):
    da = _tiff_to_xarray_nomask(ruta_qgeoide_tiff)
    region = [float(da.lon.min()), float(da.lon.max()),
              float(da.lat.min()), float(da.lat.max())]

    zmin, zmax = _zrange_from_da(da)
    cpt = _make_cpt_from_range(zmin, zmax+1e-1, cmap="jet")

    fig = pygmt.Figure()
    fig.grdimage(
        grid=da,
        region=region,
        projection="M15c",
        frame="af",
        cmap=cpt
    )
    fig.colorbar(cmap=cpt, frame=["a10", "x+lAnomalía de Altura", "y+lm"])
    fig.coast(region=region, borders="1/0.5p", shorelines="1/0.5p", frame="af")
    fig.savefig(ruta_mapa_pdf)
    fig.savefig(ruta_mapa_pdf+'.png')
    #fig.show()


# ==========================
# STD (SIN TOCAR)
# ==========================
def _tiff_to_xarray_cm(path):
    """
    Lee la banda 1 del GeoTIFF (en metros) y la convierte a centímetros (×100),
    devolviendo un xarray.DataArray con coords lon/lat.
    """
    with rasterio.open(path) as src:
        z = src.read(1, masked=False).astype(float) * 100.0  # m -> cm
        T = src.transform
        ncols, nrows = src.width, src.height

        xs = np.arange(ncols) + 0.5
        ys = np.arange(nrows) + 0.5
        lons = T.c + xs * T.a
        lats = T.f + ys * T.e

        if lats[1] < lats[0]:
            lats = lats[::-1]
            z = z[::-1, :]

        return xr.DataArray(z, coords={"lat": lats, "lon": lons}, dims=("lat", "lon"))
def mapa_nacional_std(ruta_qgeoide_tiff, ruta_mapa_pdf, grid):
    # Región nacional fija
    region = [-79, -66.8, -4.1, 13]

    # Carga en cm
    da_cm = _tiff_to_xarray_cm(ruta_qgeoide_tiff)

    fig = pygmt.Figure()
    fig.grdimage(grid=grid, region=region, projection="M15c", frame="a", cmap="geo")

    # Ploteo en cm
    fig.grdimage(
        grid=da_cm,
        region=region,
        projection="M15c",
        frame="a",
        cmap="jet",
    )


    fig.colorbar(frame=["a0.5", "x+lDesviación Estándar","y+lcm"])

    countries = [
        {"lon": -71, "lat": 4, "name": "COLOMBIA"},
        {"lon": -68, "lat": 9.0, "name": "VENEZUELA"},
        {"lon": -68, "lat": -2, "name": "BRASIL"},
        {"lon": -75, "lat": -2, "name": "PERÚ"},
        {"lon": -78, "lat": -1, "name": "ECUADOR"},
    ]
    for c in countries:
        fig.text(x=c["lon"], y=c["lat"], text=c["name"],
                 font="12p,Times-Roman,black", justify="CM")

    fig.coast(projection="M15c", borders="1/0.5p", shorelines="1/0.5p", frame="af")
    fig.savefig(ruta_mapa_pdf)
   # fig.show()


def mapa_local_std(ruta_qgeoide_tiff, ruta_mapa_pdf):
    # Carga en cm
    da_cm = _tiff_to_xarray_cm(ruta_qgeoide_tiff)
    
    zmin, zmax = _zrange_from_da(da_cm)
    
    cpt = _make_cpt_from_range(0, zmax, cmap="jet")

    # Región desde el propio raster
    region = [float(da_cm.lon.min()), float(da_cm.lon.max()),
              float(da_cm.lat.min()), float(da_cm.lat.max())]

    fig = pygmt.Figure()
    fig.grdimage(
        grid=da_cm,
        region=region,
        projection="M15c",
        frame="af",
        cmap=cpt,
    )


    fig.colorbar(frame=["a0.5", "x+lDesviación Estándar","y+lcm"])

    fig.coast(region=region, borders="1/0.5p", shorelines="1/0.5p", frame="af")
    fig.savefig(ruta_mapa_pdf)
    fig.savefig(ruta_mapa_pdf+'.png')
   # fig.show()


def creacion_mapas(ruta_qgeoide_tiff,
                   ruta_mapa_nacional_pdf,
                   ruta_mapa_local_pdf,
                   ruta_std_tiff,
                   ruta_mapa_std_nacional_pdf,
                   ruta_mapa_std_local_pdf):
    # Fondo de relieve para mapas nacionales
    grid = pygmt.datasets.load_earth_relief(resolution="15s", region=[-79, -66.8, -4.1, 13])

    # Qgeoide (ajustado)
    # mapa_nacional(ruta_qgeoide_tiff, ruta_mapa_nacional_pdf, grid)
    mapa_local(ruta_qgeoide_tiff, ruta_mapa_local_pdf)

    # STD (sin tocar)
    # mapa_nacional_std(ruta_std_tiff, ruta_mapa_std_nacional_pdf, grid)
    mapa_local_std(ruta_std_tiff, ruta_mapa_std_local_pdf)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("ruta_qgeoide_tiff", type=str, help="Ruta del modelo Qgeoide Tiff")
    parser.add_argument("ruta_mapa_nacional_pdf", type=str, help="Ruta del mapa nacional")
    parser.add_argument("ruta_mapa_local_pdf", type=str, help="Ruta del mapa local")
    parser.add_argument("ruta_std_tiff", type=str, help="Ruta de la desviación estándar modelo Qgeoide")
    parser.add_argument("ruta_mapa_std_nacional_pdf", type=str, help="Ruta del mapa nacional del STD")
    parser.add_argument("ruta_mapa_std_local_pdf", type=str, help="Ruta del mapa local del STD")
    args = parser.parse_args()

    creacion_mapas(
        ruta_qgeoide_tiff=args.ruta_qgeoide_tiff,
        ruta_mapa_nacional_pdf=args.ruta_mapa_nacional_pdf,
        ruta_mapa_local_pdf=args.ruta_mapa_local_pdf,
        ruta_std_tiff=args.ruta_std_tiff,
        ruta_mapa_std_nacional_pdf=args.ruta_mapa_std_nacional_pdf,
        ruta_mapa_std_local_pdf=args.ruta_mapa_std_local_pdf
    )