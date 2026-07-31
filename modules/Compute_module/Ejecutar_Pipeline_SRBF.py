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
    sigma_file = None,
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
    tam_bloque=5000,
    obs_filter="Shannon",
    aero_filter="Cup",
    pixel_filter="Cup",
):
    project_root = Path(project_root).resolve()

    pipeline_script = (
        project_root
        / "modules"
        / "Compute_module"
        / "5_Main"
        / "Pipeline_14072026.py"
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
        "--sigma-file",
        str(sigma_file),
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
        "--tam-bloque",
        str(tam_bloque),
        "--obs-filter",
        obs_filter,
        "--aero-filter",
        aero_filter,
        "--pixel-filter",
        pixel_filter,
    ]

    if force_steps:
        cmd.extend(["--force-steps", *[str(step) for step in force_steps]])

    if restore:
        cmd.append("--restore")
    else:
        cmd.append("--no-restore")

    if qgeoid:
        cmd.append("--qgeoid")
    else:
        cmd.append("--no-qgeoid")

    print("\nEjecutando Pipeline_14072026.py...")
    print(" ".join(f'"{x}"' if " " in x else x for x in cmd))
    print()

    subprocess.run(
        cmd,
        cwd=str(project_root),
        check=True,
    )
