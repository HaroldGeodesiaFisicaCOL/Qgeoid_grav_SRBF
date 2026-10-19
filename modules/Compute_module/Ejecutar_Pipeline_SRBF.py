import subprocess
import sys
from pathlib import Path


def ejecutar_pipeline_srbf(
    *,
    project_root,
    freqs,
    resol,
    lat_min,
    lat_max,
    lon_min,
    lon_max,
    apertura,
    puntos,
    puntos_aero,
    sigma_file=None,
    cases,
    restore=True,
    qgeoid=True,
    force_mode="none",
    force_steps=None,
    obs_lat_col="Lat_Magna",
    obs_lon_col="Long_Magna",
    obs_alt_col="h",
    aero_lat_col="Latitud",
    aero_lon_col="Longitud",
    aero_alt_col="Altura",
    pixel_lat_col="Latitud",
    pixel_lon_col="Longitud",
    pixel_alt_col="altura",
    tam_bloque=5000,
    kernel_block=2000,
    obs_filter="Shannon",
    aero_filter="Cup",
    pixel_filter="Cup",
    si_no_terr=None,
    si_no_aero=None,
    legacy=False,
):
    project_root = Path(project_root).resolve()

    pipeline_script = (
        project_root
        / "modules"
        / "Compute_module"
        / "5_Main"
        / "Pipeline_07102026.py"
    )

    if not pipeline_script.exists():
        raise FileNotFoundError(f"No existe el pipeline:\n{pipeline_script}")

    puntos = Path(puntos).resolve()
    puntos_aero = Path(puntos_aero).resolve()
    if sigma_file:
        sigma_file = Path(sigma_file).resolve()

    archivos_requeridos = {
        "puntos": puntos,
        "puntos_aero": puntos_aero,
    }
    if sigma_file:
        archivos_requeridos["sigma_file"] = sigma_file

    for nombre, ruta in archivos_requeridos.items():
        if not ruta.exists():
            raise FileNotFoundError(f"No existe {nombre}:\n{ruta}")

    force_steps = force_steps or []

    cmd = [
        sys.executable,
        "-u",
        str(pipeline_script),
        "--freq",
        *[str(freq) for freq in freqs],
        "--resol",
        str(resol),
        "--lat-min",
        str(lat_min),
        "--lat-max",
        str(lat_max),
        "--lon-min",
        str(lon_min),
        "--lon-max",
        str(lon_max),
        "--apertura",
        str(apertura),
        "--puntos",
        str(puntos),
        "--puntos-aero",
        str(puntos_aero),
        "--cases",
        *[str(case) for case in cases],
        "--force-mode",
        str(force_mode),
        "--obs-lat-col",
        obs_lat_col,
        "--obs-lon-col",
        obs_lon_col,
        "--obs-alt-col",
        obs_alt_col,
        "--aero-lat-col",
        aero_lat_col,
        "--aero-lon-col",
        aero_lon_col,
        "--aero-alt-col",
        aero_alt_col,
        "--pixel-lat-col",
        pixel_lat_col,
        "--pixel-lon-col",
        pixel_lon_col,
        "--pixel-alt-col",
        pixel_alt_col,
        "--tam-bloque",
        str(tam_bloque),
        "--kernel-block",
        str(kernel_block),
        "--obs-filter",
        obs_filter,
        "--aero-filter",
        aero_filter,
        "--pixel-filter",
        pixel_filter,
    ]

    # Solo se pasa si se indicó; de lo contrario el pipeline usa su valor por defecto
    if sigma_file:
        cmd.extend(["--sigma-file", str(sigma_file)])

    if si_no_terr is not None:
        cmd.extend(["--si-no-terr", str(si_no_terr)])
    if si_no_aero is not None:
        cmd.extend(["--si-no-aero", str(si_no_aero)])

    if force_steps:
        cmd.extend(["--force-steps", *[str(step) for step in force_steps]])

    cmd.append("--restore" if restore else "--no-restore")
    cmd.append("--qgeoid" if qgeoid else "--no-qgeoid")

    # Modo legacy: calcula la matriz de diseño de píxeles en disco (.npy).
    # Por defecto el pipeline trabaja en modo in-situ.
    if legacy:
        cmd.append("--legacy")

    print("\nEjecutando Pipeline_07102026.py...")
    print(" ".join(f'"{x}"' if " " in x else x for x in cmd))
    print()

    subprocess.run(
        cmd,
        cwd=str(project_root),
        check=True,
    )
