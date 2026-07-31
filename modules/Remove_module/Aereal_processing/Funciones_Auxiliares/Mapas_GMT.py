import pygmt
import pandas as pd
from shapely.geometry import Polygon,Point


def Mapa(region,df1,df2,Ruta_mapa):
    fig1 = pygmt.Figure()
    fig1.coast(
        region=region,          # Especificar la región
        shorelines='1/0.05p',
        projection="M15c",      # Proyección Mercator
        land="lightgray",       # Color para las áreas terrestres
        water="lightblue",      # Color para las áreas de agua
        borders="1/0.5p,black", # Fronteras políticas
        frame="ag",             # Marco automático con anotaciones
    )
    fig1.plot(
        region=region,
        x=df1["lon"],
        y=df1["lat"],
        style="c0.03c",  # Círculos de 0.2 cm
        fill="darkblue",
        label="Observaciones Limpios",
    )
    fig1.plot(
        region=region,
        x=df2["lon"],
        y=df2["lat"],
        style="c0.03c",  # Círculos de 0.2 cm
        fill="red",
        label="Observaciones Atípicas",
    )
    fig1.savefig(Ruta_mapa,dpi=720)
    fig1.show()

def Mapa_valor_gravedad(region,df,col_lat,col_lon, Ruta_Mapa,zmin,zmax, Columna):

    # Crear la CPT y guardarla en un archivo temporal
    cpt_file = "colormap.cpt"
    pygmt.makecpt(cmap="jet", series=[zmin, zmax, (zmax - zmin) / 100], output=cpt_file)

    fig = pygmt.Figure()
    fig.coast(
        region=region,
        shorelines='1/0.05p',
        projection="M15c",
        land="lightgray",
        # water="lightblue",
        borders="1/0.5p,black",
        frame="af",
    )
    fig.plot(
        x=df[col_lon],
        y=df[col_lat],
        fill=df[Columna],
        style="c0.05c",
        cmap=cpt_file  # Usa la CPT explícitamente
    )
    fig.colorbar(cmap=cpt_file, frame='af+lPerturbaciones de Gravedad')  # Asocia la barra de color a la CPT

    fig.savefig(Ruta_Mapa, dpi=600)
    fig.show()
