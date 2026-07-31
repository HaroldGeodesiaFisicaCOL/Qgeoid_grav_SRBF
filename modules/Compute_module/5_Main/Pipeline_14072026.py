import sys
import os
import io
import platform
import subprocess
import logging
import argparse
import contextlib
import runpy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple, Sequence
import shutil
import pandas as pd
from monitor_recursos1 import MonitorRecursos
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s: %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger("pipeline_robusto")

BASE_DIR = Path(__file__).resolve().parent.parent

def R(*parts) -> Path:
    return (BASE_DIR.joinpath(*parts)).resolve()

DEFAULT_INPUTS = {
    "puntos": R("3_Observaciones", "Obs_Terr_DTU21Gra_Nosotros.txt"),
    "puntos_aero": R("3_Observaciones", "Aero_grav_ob_en_IHRF_MA_2000.txt"),
    "listado": R("5_Main", "listado.txt"),
}
DEFAULT_SIGMA_FILE = R("3_Observaciones", "Informe_varianza_x_proyectos_f2190.txt")
DEFAULT_FREQ = "5700"
DEFAULT_RESOL = "1"
DEFAULT_LAT_MAX = "9.0427396123999593"
DEFAULT_LAT_MIN = "3.3594065063999592"
DEFAULT_LON_MAX = "-72.7355829984999929"
DEFAULT_LON_MIN = "-78.4189161044999992"
DEFAULT_APERTURA = "0.326666666666666666666666666666"
DEFAULT_SINO = "No"
DEFAULT_BLOCK_SIZE = 5000
DEFAULT_CASES = ["1VC"]

CASE_DEFINITIONS = {
    "1VC": (True, False),
    "1VCLC": (True, True),
    "SigmasVC": (False, False),
    "SigmasVCLC": (False, True),
}

@dataclass
class DesignColumns:
    obs_lat: str = "Lat_Magna"
    obs_lon: str = "Long_Magna"
    obs_alt: str = "h"
    aero_lat: str = "Latitud"
    aero_lon: str = "Longitud"
    aero_alt: str = "Altura"
    block_size: int = DEFAULT_BLOCK_SIZE

@dataclass
class DesignFilters:
    obs_filter: str = "Shannon"
    aero_filter: str = "Cup"
    pixel_filter: str = "Cup"

@dataclass
class PipelineConfig:
    freqs: List[str] = field(default_factory=lambda: [DEFAULT_FREQ])
    resol: str = DEFAULT_RESOL
    lat_max: str = DEFAULT_LAT_MAX
    lat_min: str = DEFAULT_LAT_MIN
    lon_max: str = DEFAULT_LON_MAX
    lon_min: str = DEFAULT_LON_MIN
    apertura: str = DEFAULT_APERTURA
    si_no: str = DEFAULT_SINO
    puntos: Path = DEFAULT_INPUTS["puntos"]
    puntos_aero: Path = DEFAULT_INPUTS["puntos_aero"]
    listado: Path = DEFAULT_INPUTS["listado"]
    sigma_file: Path = DEFAULT_SIGMA_FILE
    cases: List[str] = field(default_factory=lambda: list(DEFAULT_CASES))
    force_mode: str = "none"  # none | all | steps
    force_steps: List[str] = field(default_factory=list)
    restore: bool = True
    qgeoid: bool = True
    design_cols: DesignColumns = field(default_factory=DesignColumns)
    design_filters: DesignFilters = field(default_factory=DesignFilters)

scripts = {
    "reuter":         R("4_Rutinas_encerradas", "ReuterGrid.py"),
    "diseno":         R("4_Rutinas_encerradas", "Calc_puntual_optimizado_partes.py"),
    "graflab_path":   R("4_Rutinas_encerradas"),
    "graflab_expand": R("4_Rutinas_encerradas", "GravLab_expand.py"),
    "vce":            R("4_Rutinas_encerradas", "VCE06042026C.py"),
    "pixeles":        R("4_Rutinas_encerradas", "generador_pixel.py"),
    "interpolado":    R("4_Rutinas_encerradas", "interpolado_modelos_pixeles.py"),
    "raster":         R("4_Rutinas_encerradas", "Generador_de_Raster1.py"),
    "restore":        R("4_Rutinas_encerradas", "Restore1.py"),
    "qgeoid":         R("4_Rutinas_encerradas", "QGeoid1.py"),
    "std":            R("4_Rutinas_encerradas", "Desv_Standar_GPU_Mejorado_241225.py"),
    "visu":           R("4_Rutinas_encerradas", "Visualizacion.py"),
    "lcurve":         R("4_Rutinas_encerradas", "LC060.py"),
    "Sign_Param":     R("4_Rutinas_encerradas", "Significancia_Parametros.py"),
    "raster_STD_T":   R("4_Rutinas_encerradas", "Desviacion_Potencial.py"),
    "zero_zeta":      R("4_Rutinas_encerradas", "termino_orden_cero.py"),
}

modelos_base = {
    "srtm30":  R("1_Modelos", "modelo_SRTM", "SRTM_v7_Plus.tif"),
    "EGM2008": R("1_Modelos", "modelo_EGM2008", "EGM2008.tif"),
    "GGM":     R("1_Modelos", "modeloXGM2019", "XGM2019.gfc"),
    "Topo":    R("1_Modelos", "modelo_dv_ell_Earth2014", "dV_ELL_Earth2014_5480_plusGRS80.gfc"),
    "ERTM":    R("1_Modelos", "modeloERTM2160", "Anomalias_Altura_ERTM_2160.tif"),
}

outputs_base = {
    "reuter_grid_local": R("2_Reuter_Grid", "Reuter_Grid.txt"),
    "Restore_dir":             R("6_Resultados", "Restore"),
    "Expand_GGM":              R("6_Resultados", "XGM2019_0_719"),
    "Expand_topo1":            R("6_Resultados", "dv_ell_earth2014_0_719"),
    "Expand_topo2":            R("6_Resultados", "dv_ell_earth2014_0_2159"),
    "puntos_centrales_matlab": R("6_Resultados", "puntos_centrales_matlab.txt"),
    "matriz_diseno":      R("6_Resultados", "matriz_diseno.npy"),
    "matriz_diseno_aero":  R("6_Resultados", "matriz_diseno_aero.npy"),
    "matriz_pixel":        R("6_Resultados", "matriz_diseno_pixel.npy"),
    "pixeles_txt":         R("6_Resultados", "raster_puntos_centrales.txt"),
    "param_est":  R("6_Resultados", "parametros_estimados.txt"),
    "pesos":      R("6_Resultados", "pesos.csv"),
    "residuos":    R("6_Resultados", "errores_obs.txt"),
    "lc_residuals":    R("6_Resultados", "errores_obs.txt"),
    "normas":      R("6_Resultados", "Normas.txt"),
    "lc_params":   R("6_Resultados", "parametros_optimizados.txt"),
    "lc_cov":      R("6_Resultados", "matriz_varianza_covarianza.npy"),
    "raster_T":     R("6_Resultados", "Potencial_Perturbador.tif"),
    "raster_T_std": R("6_Resultados", "Potencial_Perturbador_STD.tif"),
    "raster_std":   R("6_Resultados", "Std.tif"),
    "restore_txt":  R("6_Resultados", "Restore.txt"),
    "raster_Z":     R("6_Resultados", "Qgeoide_Gravimetrico.tif"),
    "mapa_nac":       R("7_Visualizaciones", "Qgeoide_Gravimetrico.pdf"),
    "mapa_loc":       R("7_Visualizaciones", "Qgeoide_Gravimetrico_Delimitado.pdf"),
    "mapa_std_nac":   R("7_Visualizaciones", "STD.pdf"),
    "mapa_std_loc":   R("7_Visualizaciones", "STD_Delimitado.pdf"),
    "mapa_T":         R("7_Visualizaciones", "Potencial_Perturbador.pdf"),
    "mapa_std_T":     R("7_Visualizaciones", "Mapa_Potencial_T_STD_local.pdf"),
    "mapa_param_sig": R("7_Visualizaciones", "Mapa_parametros_sign.pdf"),
    "mapa_param_sig_s": R("7_Visualizaciones", "Mapa_parametros_sign_solos.pdf"),
    "mapa_param":     R("7_Visualizaciones", "Mapa_parametros.pdf"),
    "mapa_param_STD": R("7_Visualizaciones", "Mapa_STD_Parametros.pdf"),
    "sing_param":     R("7_Visualizaciones", "significancia_parametros.txt"),
    "t_estats":       R("7_Visualizaciones", "t_stats_parametros.txt"),
    "histo_t_stats":  R("7_Visualizaciones", "Ploteo_t_stats.pdf"),
    "histo_params":   R("7_Visualizaciones", "Ploteo_param.pdf"),
}

def resolve_input_path(p: Path, folder: str) -> Path:
    p = Path(p)
    if p.is_absolute():
        return p
    if len(p.parts) == 1:   # solo nombre de archivo
        return R(folder, p.name)
    return (BASE_DIR / p).resolve()


def ensure_parent(p: Path):
    if p is not None:
        Path(p).parent.mkdir(parents=True, exist_ok=True)

def ensure_directory(p: Path):
    if p is not None:
        Path(p).mkdir(parents=True, exist_ok=True)

def suffix_path_case(p: Path, freq: str, case: str, carpeta: str = None) -> Path:
    if p is None:
        return p
    p = Path(p)
    new_name = f"{p.stem}_f{freq}{p.suffix}"
    if carpeta is not None:
        parent = BASE_DIR / carpeta / f"f{freq}" / case
    else:
        if "6_Resultados" in str(p):
            parent = BASE_DIR / "6_Resultados" / f"f{freq}" / case
        elif "7_Visualizaciones" in str(p):
            parent = BASE_DIR / "7_Visualizaciones" / f"f{freq}" / case
        else:
            parent = p.parent
    parent.mkdir(parents=True, exist_ok=True)
    return parent / new_name

def candidate_paths(p: Path) -> List[Path]:
    p = Path(p)
    cands = [p]
    if p.suffix == "":
        cands.append(p.with_suffix(".txt"))
    return cands

def file_ready(p: Path) -> bool:
    if p is None:
        return False
    for cand in candidate_paths(Path(p)):
        if cand.exists() and cand.is_file() and cand.stat().st_size > 0:
            return True
    return False

def outputs_ready(paths: List[Path]) -> bool:
    if not paths:
        return False
    return all(file_ready(p) for p in paths)

def normalize_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "si", "sí", "on"}
    return bool(value)

def force_matches(step_key: str, patterns: Sequence[str]) -> bool:
    step_key = step_key.lower().strip()
    for pattern in patterns:
        pattern = str(pattern).lower().strip()
        if not pattern:
            continue
        if step_key == pattern or step_key.startswith(pattern) or pattern.startswith(step_key):
            return True
    return False

def should_force_step(config: PipelineConfig, step_key: str, always: bool = False) -> bool:
    if always:
        return True
    if config.force_mode == "all":
        return True
    if config.force_mode == "steps":
        return force_matches(step_key, config.force_steps)
    return False

def run_subprocess_capture(script: Path, descripcion: str, args: List[str], env=None, allow_runpy_if_long: bool = True) -> Tuple[int, List[str]]:
    script = Path(script)
    args = [str(a) for a in args]
    cmd = [sys.executable, "-u", str(script), *args]
    cmd_len = sum(len(x) + 1 for x in cmd)
    should_use_runpy = allow_runpy_if_long and platform.system().lower().startswith("win") and cmd_len > 7000
    if should_use_runpy:
        logger.info(f"[RUNPY] {descripcion} (cmd_len={cmd_len}) -> {script}")
        old_argv = sys.argv.copy()
        old_cwd = os.getcwd()
        try:
            sys.argv = [str(script), *args]
            os.chdir(script.parent)
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
                try:
                    runpy.run_path(str(script), run_name="__main__")
                    out = buf.getvalue().splitlines()
                    for line in out:
                        print(line, flush=True)
                    return 0, out
                except SystemExit as e:
                    out = buf.getvalue().splitlines()
                    code = int(e.code) if isinstance(e.code, int) else 1
                    for line in out:
                        print(line, flush=True)
                    return code, out
                except Exception as e:
                    out = buf.getvalue().splitlines()
                    err_line = f"[EXCEPTION] {descripcion}: {e}"
                    logger.exception(err_line)
                    out.append(err_line)
                    for line in out:
                        print(line, flush=True)
                    return 1, out
        finally:
            sys.argv = old_argv
            os.chdir(old_cwd)
    logger.info(f"Ejecutando subprocess: {descripcion}")
    try:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, universal_newlines=True, env=env)
        out_lines: List[str] = []
        for line in proc.stdout:
            if line:
                print(line, end="", flush=True)
                out_lines.append(line.rstrip())
        proc.stdout.close()
        rc = proc.wait()
        if rc != 0:
            logger.error(f"[ERROR] Falló {descripcion} (code {rc})")
        return rc, out_lines
    except Exception as e:
        logger.exception(f"[ERROR] Excepción {descripcion}: {e}")
        return 1, [f"EXCEPTION: {e}"]

def run_step(name: str, script: Path, args: List[str], outputs: List[Path], force: bool, env=None) -> Tuple[int, List[str]]:
    if (not force) and outputs_ready(outputs):
        msg = f"[SKIP] {name} (ya existe)"
        logger.info(msg)
        return 0, [msg]
    return run_subprocess_capture(script=script, descripcion=name, args=args, env=env)

def crear_reporte(resumen: List[str], ruta_salida: Path):
    ruta_salida.parent.mkdir(parents=True, exist_ok=True)
    with ruta_salida.open("w", encoding="utf-8") as f:
        f.write("REPORTE DE CÁLCULO\n")
        f.write("=================================\n\n")
        for linea in resumen:
            f.write(linea + "\n")

def crear_listado_automatico(listado_path: Path, outputs_common: Dict[str, Path], input_paths: Dict[str, Path], si_no: str, rel_paths: bool = False):
    listado_path.parent.mkdir(parents=True, exist_ok=True)
    def _fmt(p: Path):
        p = Path(p)
        return p.relative_to(BASE_DIR) if rel_paths else p.resolve()
    lines = []
    if "matriz_diseno" in outputs_common and "puntos" in input_paths:
        lines.append(f"{_fmt(outputs_common['matriz_diseno'])}\t{_fmt(input_paths['puntos'])}\t No")
    if "matriz_diseno_aero" in outputs_common and "puntos_aero" in input_paths:
        lines.append(f"{_fmt(outputs_common['matriz_diseno_aero'])}\t{_fmt(input_paths['puntos_aero'])}\t{si_no}")
    with listado_path.open("w", encoding="utf-8") as f:
        for l in lines:
            f.write(l + "\n")
    logger.info(f"Listado creado: {listado_path} ({len(lines)} entradas)")

def construir_sigma2_init_list(listado_path: Path, use_ones: bool, sigma_file: Path) -> str:
    listado_path = Path(listado_path)
    if not listado_path.exists():
        raise FileNotFoundError(listado_path)
    sigmas_map: Dict[str, float] = {}
    sigma_file = Path(sigma_file)
    if not use_ones:
        if not sigma_file.exists():
            raise FileNotFoundError(f"No existe sigma_file y use_ones=False: {sigma_file}")
        with sigma_file.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split("\t") if "\t" in line else line.split()
                if len(parts) < 2:
                    continue
                try:
                    sigmas_map[parts[0].strip()] = float(parts[1])
                except Exception:
                    sigmas_map[parts[0].strip()] = 1.0
    sigma_resultado: List[str] = []
    with listado_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            cols = line.split("\t")
            if len(cols) < 3:
                continue
            ruta_obs, flag = cols[1], cols[2].strip().lower()
            df_obs = pd.read_csv(ruta_obs, sep="\t")
            if "Proyecto" not in df_obs.columns:
                raise ValueError(f"No existe la columna 'Proyecto' en {ruta_obs}")
            if flag == "si":
                for proj in df_obs["Proyecto"].unique():
                    sigma_resultado.append("1" if use_ones else str(sigmas_map.get(proj, 1.0)))
            else:
                sigma_resultado.append("1")
    return ",".join(sigma_resultado) if sigma_resultado else "1"

# OUTPUTS

def build_outputs_restore_pixel() -> Dict[str, Path]:
    base = BASE_DIR / "6_Resultados" / "Restore_Pixel"
    base.mkdir(parents=True, exist_ok=True)
    return {
        "Restore_dir": base / "Restore",
        "pixeles_txt": base / "raster_puntos_centrales.txt",
        "Expand_GGM": base / "XGM2019_0_719",
        "Expand_topo1": base / "dv_ell_earth2014_0_719",
        "Expand_topo2": base / "dv_ell_earth2014_0_2159",
        "puntos_centrales_matlab": base / "puntos_centrales_matlab.txt",
    }

def build_outputs_common(freq: str) -> Dict[str, Path]:
    (BASE_DIR / "6_Resultados" / f"f{freq}" / "COMMON").mkdir(parents=True, exist_ok=True)
    (BASE_DIR / "7_Visualizaciones" / f"f{freq}" / "COMMON").mkdir(parents=True, exist_ok=True)
    out: Dict[str, Path] = {}
    out["reuter_grid_local"] = suffix_path_case(outputs_base["reuter_grid_local"], freq, "COMMON", carpeta=None)
    for key in ["matriz_diseno", "matriz_diseno_aero", "matriz_pixel"]:
        out[key] = suffix_path_case(outputs_base[key], freq, "COMMON", carpeta=None)
    return out

def build_outputs_case(freq: str, case_name: str) -> Dict[str, Path]:
    out: Dict[str, Path] = {}
    out["pixeles_txt"] = suffix_path_case(outputs_base["pixeles_txt"], freq, case_name, carpeta="6_Resultados")
    for key in ["param_est", "pesos", "residuos", "normas", "lc_params", "lc_cov","lc_residuals"]:
        out[key] = suffix_path_case(outputs_base[key], freq, case_name, carpeta="6_Resultados")
    for key in ["raster_T", "raster_T_std", "raster_std", "restore_txt", "raster_Z"]:
        out[key] = suffix_path_case(outputs_base[key], freq, case_name, carpeta="6_Resultados")
    for key in ["mapa_nac", "mapa_loc", "mapa_std_nac", "mapa_std_loc", "mapa_T", "mapa_std_T", "mapa_param_sig", "mapa_param_sig_s", "mapa_param", "mapa_param_STD", "sing_param", "t_estats", "histo_t_stats", "histo_params"]:
        out[key] = suffix_path_case(outputs_base[key], freq, case_name, carpeta="7_Visualizaciones")
    out["report_case"] = BASE_DIR / "6_Resultados" / f"f{freq}" / case_name / f"Reporte_Caso_{case_name}_f{freq}.txt"
    return out

# PASOS COMUNES

def run_restore_pixel(lat_max: str, lat_min: str, lon_max: str, lon_min: str, resol: str, outputs_rp: Dict[str, Path], force: bool = False) -> List[str]:
    ensure_directory(outputs_rp["Restore_dir"])
    for key, p in outputs_rp.items():
        if key != "Restore_dir":
            ensure_parent(p)
    steps = [
        {"name": "Generar pixeles", "key": "restore_pixel_pixeles", "script": scripts["pixeles"], "args": [lat_max, lat_min, lon_max, lon_min, resol, str(outputs_rp["pixeles_txt"])], "outputs": [outputs_rp["pixeles_txt"]]},
        {"name": "Interpolado modelos en píxeles", "key": "restore_pixel_interpolado", "script": scripts["interpolado"], "args": [str(outputs_rp["pixeles_txt"]), str(modelos_base["srtm30"]), str(modelos_base["EGM2008"])], "outputs": [outputs_rp["Expand_GGM"], outputs_rp["Expand_topo1"], outputs_rp["Expand_topo2"], outputs_rp["puntos_centrales_matlab"]]},
        {"name": "GravLab expand", "key": "restore_pixel_expand", "script": scripts["graflab_expand"], "args": [str(outputs_rp["Restore_dir"]), str(scripts["graflab_path"]), str(outputs_rp["pixeles_txt"]), str(outputs_rp["puntos_centrales_matlab"]), str(modelos_base["GGM"]), str(outputs_rp["Expand_GGM"]), str(modelos_base["Topo"]), str(outputs_rp["Expand_topo1"]), str(outputs_rp["Expand_topo2"])], "outputs": [outputs_rp["Expand_GGM"], outputs_rp["Expand_topo1"], outputs_rp["Expand_topo2"], outputs_rp["puntos_centrales_matlab"]]},
    ]
    resumen: List[str] = []
    for step in steps:
        resumen.append(f"--- {step['name']} ---")
        rc, out = run_step(step["name"], step["script"], step["args"], step.get("outputs", []), force)
        resumen.extend(out)
        if rc != 0:
            raise RuntimeError(f"Falló el paso global '{step['name']}' (code {rc})")
        resumen.append("")
    return resumen

def build_common_steps(freq: str, resol: str, lat_max: str, lat_min: str, lon_max: str, lon_min: str, apertura: str, design_cols: DesignColumns, design_filters: DesignFilters, outputs_common: Dict[str, Path], outputs_rp: Dict[str, Path], input_paths: Dict[str, Path]):
    lat_max_ext = str(float(lat_max) + float(apertura))
    lat_min_ext = str(float(lat_min) - float(apertura))
    lon_max_ext = str(float(lon_max) + float(apertura))
    lon_min_ext = str(float(lon_min) - float(apertura))
    return [
        {
            "name": "Reuter Grid",
            "script": scripts["reuter"],
            "args": [str(freq), str(outputs_common["reuter_grid_local"]), lat_max_ext, lat_min_ext, lon_max_ext, lon_min_ext],
            "outputs": [outputs_common["reuter_grid_local"]],
        },
        {
       "name": "Matriz diseño (observaciones)",
       "key": "diseno_obs",
       "script": scripts["diseno"],
       "args": [
           str(input_paths["puntos"]),                 # ruta_puntos
           design_cols.obs_lat,                       # col_lat
           design_cols.obs_lon,                       # col_lon
           design_cols.obs_alt,                       # col_alt
           str(outputs_common["reuter_grid_local"]),  # ruta_malla
           "Latitud",                                 # col_lat_malla
           "Longitud",                                # col_lon_malla
           str(freq),                                 # num_filas
           "0",                                       # para_pixeles
           str(outputs_common["matriz_diseno"]),      # ruta_salida
           design_filters.obs_filter,                 # filtro
           "--tam_bloque",
           str(design_cols.block_size),
       ],
       "outputs": [outputs_common["matriz_diseno"]],
   },

   {
       "name": "Matriz diseño (observaciones_aereo)",
       "key": "diseno_aero",
       "script": scripts["diseno"],
       "args": [
           str(input_paths["puntos_aero"]),              # ruta_puntos
           design_cols.aero_lat,                        # col_lat
           design_cols.aero_lon,                        # col_lon
           design_cols.aero_alt,                        # col_alt
           str(outputs_common["reuter_grid_local"]),    # ruta_malla
           "Latitud",                                   # col_lat_malla
           "Longitud",                                  # col_lon_malla
           str(freq),                                   # num_filas
           "0",                                         # para_pixeles
           str(outputs_common["matriz_diseno_aero"]),   # ruta_salida
           design_filters.aero_filter,                  # filtro
           "--tam_bloque",
           str(design_cols.block_size),
       ],
       "outputs": [outputs_common["matriz_diseno_aero"]],
   },

   {
       "name": "Matriz diseño (pixeles)",
       "key": "diseno_pixeles",
       "script": scripts["diseno"],
       "args": [
           str(outputs_rp["pixeles_txt"]),           # ruta_puntos
           "Latitud",                                # col_lat
           "Longitud",                               # col_lon
           "altura",                                 # col_alt
           str(outputs_common["reuter_grid_local"]), # ruta_malla
           "Latitud",                                # col_lat_malla
           "Longitud",                               # col_lon_malla
           str(freq),                                # num_filas
           "1",                                      # para_pixeles
           str(outputs_common["matriz_pixel"]),      # ruta_salida
           design_filters.pixel_filter,              # filtro
           "--tam_bloque",
           str(design_cols.block_size),
       ],
       "outputs": [outputs_common["matriz_pixel"]],
   },
    ]

def run_common(freq: str, resol: str, lat_max: str, lat_min: str, lon_max: str, lon_min: str, apertura: str, si_no: str, force: bool, outputs_rp: Dict[str, Path], input_paths: Dict[str, Path], design_cols: DesignColumns, design_filters: DesignFilters):
    outputs_common = build_outputs_common(freq)
    for p in outputs_common.values():
        ensure_parent(p)
    crear_listado_automatico(input_paths["listado"], outputs_common, input_paths, si_no, rel_paths=False)
    steps = build_common_steps(freq, resol, lat_max, lat_min, lon_max, lon_min, apertura, design_cols, design_filters, outputs_common, outputs_rp, input_paths)
    resumen: List[str] = []
    for step in steps:
        resumen.append(f"--- {step['name']} ---")
        rc, out = run_step(step["name"], step["script"], step["args"], step.get("outputs", []), force)
        resumen.extend(out)
        if rc != 0:
            raise RuntimeError(f"Falló el paso común '{step['name']}' (code {rc})")
        resumen.append("")
    return outputs_common, resumen

# VCE / LC

def run_shared_vce(freq: str, use_ones: bool, sigma_file: Path, input_paths: Dict[str, Path], force: bool = False, boolCov: bool = True) -> Tuple[Dict[str, Path], List[str]]:
    tag = "ones" if use_ones else "sigmas"
    shared_dir = BASE_DIR / "6_Resultados" / f"f{freq}" / "COMMON" / f"VCE_shared_{tag}"
    shared_dir.mkdir(parents=True, exist_ok=True)
    shared_outputs = {"param_est": shared_dir / f"parametros_estimados_{tag}_f{freq}.txt", "pesos": shared_dir / f"pesos_{tag}_f{freq}.csv", "residuos": shared_dir / f"errores_obs_{tag}_f{freq}.txt", "normas": shared_dir / f"Normas_{tag}_f{freq}.txt"}
    if boolCov:
        shared_outputs["vce_cov"] = shared_dir / f"matriz_varianza_covarianza_{tag}_f{freq}.npy"
    if (not force) and outputs_ready(list(shared_outputs.values())):
        msg = f"[SKIP] VCE_shared_{tag} (ya existe)"
        logger.info(msg)
        return shared_outputs, [msg]
    sigma2_init_list = construir_sigma2_init_list(input_paths["listado"], use_ones, sigma_file)
    vce_args = ["--rutas_archivo","--col_val" ,str(input_paths["listado"]), "--sigma2_init_list", sigma2_init_list, "--sigma_mu2_init", "1.0", "--max_iter", "50", "--tol", "1e-11", "--num_trace_samples", "1", "--output_beta", str(shared_outputs["param_est"]), "--output_weights", str(shared_outputs["pesos"]), "--output_residuales", str(shared_outputs["residuos"]), "--output_normas", str(shared_outputs["normas"])]
    if boolCov:
        vce_args.extend(["--output_covariance", str(shared_outputs["vce_cov"])])
    rc, out = run_subprocess_capture(scripts["vce"], f"VCE_shared_{tag}", vce_args)
    if rc != 0:
        raise RuntimeError(f"[Shared VCE {tag}] falló (code {rc})")
    return shared_outputs, out

def run_shared_lcurve(freq: str, use_ones: bool, shared_vce: Dict[Tuple[bool, bool], Dict[str, Path]], input_paths: Dict[str, Path], force: bool = False) -> Tuple[Dict[str, Path], List[str]]:
    tag = "ones" if use_ones else "sigmas"
    shared_dir = BASE_DIR / "6_Resultados" / f"f{freq}" / "COMMON" / f"LC_shared_{tag}"
    shared_dir.mkdir(parents=True, exist_ok=True)
    shared_outputs = {
    "lc_params": shared_dir / f"parametros_optimizados_{tag}_f{freq}.txt",
    "lc_cov": shared_dir / f"matriz_varianza_covarianza_{tag}_f{freq}.npy",
    "lc_residuals": shared_dir / f"errores_obs_{tag}_f{freq}.txt",
    "ReporLC": shared_dir / f"Reporte_L_Curva_{tag}_f{freq}.pdf"}
    if (not force) and outputs_ready(list(shared_outputs.values())):
        msg = f"[SKIP] LC_shared_{tag} (ya existe)"
        logger.info(msg)
        return shared_outputs, [msg]
    if (use_ones, False) in shared_vce:
        vce_out = shared_vce[(use_ones, False)]
    elif (use_ones, True) in shared_vce:
        vce_out = shared_vce[(use_ones, True)]
    else:
        raise KeyError(f"No hay resultados VCE para use_ones={use_ones}")
    if not Path(vce_out["pesos"]).exists() or not Path(vce_out["normas"]).exists():
        raise FileNotFoundError("Faltan pesos/normas para ejecutar LC compartido.")
    lc_args = [
    "--listado", str(input_paths["listado"]),
    "--pesos", str(vce_out["pesos"]),
    "--normas", str(vce_out["normas"]),
    "--out_params", str(shared_outputs["lc_params"]),
    "--out_cov", str(shared_outputs["lc_cov"]),
    "--out_residuals", str(shared_outputs["lc_residuals"]),
    "--out_fig", str(shared_outputs["ReporLC"])]
    rc, out = run_subprocess_capture(scripts["lcurve"], f"LC_shared_{tag}", lc_args)
    if rc != 0:
        raise RuntimeError(f"[Shared LC {tag}] falló (code {rc})")
    return shared_outputs, out

# CASOS

def run_case(freq: str, case_name: str, use_ones: bool, do_lc: bool, outputs_common: Dict[str, Path], restore_pixel_outputs: Dict[str, Path], shared_vce_outputs: Dict[Tuple[bool, bool], Dict[str, Path]], shared_lc_outputs: Dict[bool, Dict[str, Path]], resol: str, force_mode: str, force_steps: Sequence[str], input_paths: Dict[str, Path], activate_restore: bool = False, activate_qgeoid: bool = True):
    logger.info(f"=== Ejecutando caso {case_name} (freq={freq}) ===")
    outputs_case = build_outputs_case(freq, case_name)
    for p in outputs_case.values():
        ensure_parent(p)
    resumen: List[str] = []
    if (use_ones, True) in shared_vce_outputs:
        shared_vce = shared_vce_outputs[(use_ones, True)]
    elif (use_ones, False) in shared_vce_outputs:
        shared_vce = shared_vce_outputs[(use_ones, False)]
    else:
        raise RuntimeError(f"No hay VCE disponible para use_ones={use_ones}")
    shared_lc = shared_lc_outputs.get(use_ones, {}) if do_lc else {}
    if do_lc and shared_lc:
        params_for_raster = shared_lc["lc_params"]
        cov_for_std = shared_lc["lc_cov"]
        params_for_sign = shared_lc["lc_params"]
        cov_for_sign = shared_lc["lc_cov"]
    else:
        params_for_raster = shared_vce["param_est"]
        cov_for_std = shared_vce["vce_cov"]
        params_for_sign = shared_vce["param_est"]
        cov_for_sign = shared_vce["vce_cov"]
    pixeles_case = outputs_case["pixeles_txt"]
    shutil.copy2(restore_pixel_outputs["pixeles_txt"], pixeles_case)
    def step_force(key: str, always: bool = False) -> bool:
        if always:
            return True
        if force_mode == "all":
            return True
        if force_mode == "steps":
            return force_matches(key, force_steps)
        return False
    raster_args = [str(outputs_common["matriz_pixel"]), str(params_for_raster), str(pixeles_case), str(outputs_case["raster_T"]), "T"]
    rc, out = run_step(f"Raster T ({case_name})", scripts["raster"], raster_args, [outputs_case["raster_T"]], step_force("raster_t"))
    resumen.extend(out)
    if rc != 0:
        raise RuntimeError(f"[{case_name}] Raster T falló (code {rc})")
    resumen.append("")
    if activate_restore:
        restore_args = [str(pixeles_case), str(restore_pixel_outputs["Expand_GGM"]), str(restore_pixel_outputs["Expand_topo1"]), str(restore_pixel_outputs["Expand_topo2"]), str(modelos_base["ERTM"]), str(outputs_case["restore_txt"]), "1e-6", "50"]
        rc, out = run_step(f"Restore ({case_name})", scripts["restore"], restore_args, [outputs_case["restore_txt"]], step_force("restore"))
        resumen.extend(out)
        if rc != 0:
            raise RuntimeError(f"[{case_name}] Restore falló (code {rc})")
        resumen.append("")
        std_args = [str(cov_for_std), str(outputs_case["restore_txt"]), str(outputs_common["matriz_pixel"]), str(outputs_case["raster_std"]), resol, "--batch_size", "150000", "--tile_m", "8192", "--num_workers", "1"]
        rc, out = run_step(f"Desviación estándar residual ({case_name})", scripts["std"], std_args, [outputs_case["raster_std"]], step_force("std"))
        resumen.extend(out)
        if rc != 0:
            raise RuntimeError(f"[{case_name}] STD falló (code {rc})")
        resumen.append("")
        if activate_qgeoid:
            qgeoid_args = [str(outputs_case["restore_txt"]), str(outputs_case["raster_Z"]), resol]
            rc, out = run_step(f"QGeoid ({case_name})", scripts["qgeoid"], qgeoid_args, [outputs_case["raster_Z"]], True)
            resumen.extend(out)
            if rc != 0:
                raise RuntimeError(f"[{case_name}] QGeoid falló (code {rc})")
            resumen.append("")
            visu_args = [str(outputs_case["raster_Z"]), str(outputs_case["mapa_nac"]), str(outputs_case["mapa_loc"]), str(outputs_case["raster_std"]), str(outputs_case["mapa_std_nac"]), str(outputs_case["mapa_std_loc"])]
            rc, out = run_step(f"Visualización mapas ({case_name})", scripts["visu"], visu_args, [outputs_case["mapa_std_nac"], outputs_case["mapa_std_loc"], outputs_case["mapa_nac"], outputs_case["mapa_loc"]], step_force("visu"))
            resumen.extend(out)
            if rc != 0:
                raise RuntimeError(f"[{case_name}] Visualización falló (code {rc})")
            resumen.append("")
    else:
        logger.info(f"[{case_name}] Restore desactivado: se omiten STD, QGeoid y Visualización.")
    sp_args = [str(params_for_sign), str(input_paths["listado"]), str(cov_for_sign), str(outputs_case["t_estats"]), str(outputs_case["histo_params"]), str(outputs_case["histo_t_stats"]), str(outputs_case["sing_param"]), str(outputs_common["reuter_grid_local"]), str(outputs_case["mapa_param_sig"]), str(outputs_case["mapa_param_sig_s"]), str(outputs_case["mapa_param"]), str(outputs_case["mapa_param_STD"])]
    rc, out = run_step(f"Significancia ({case_name})", scripts["Sign_Param"], sp_args, [outputs_case["t_estats"], outputs_case["histo_params"], outputs_case["histo_t_stats"], outputs_case["sing_param"], outputs_case["mapa_param_sig"], outputs_case["mapa_param_sig_s"], outputs_case["mapa_param"], outputs_case["mapa_param_STD"]], step_force("significancia"))
    resumen.extend(out)
    if rc != 0:
        raise RuntimeError(f"[{case_name}] Significancia falló (code {rc})")
    resumen.append("")
    if Path(outputs_case["restore_txt"]).exists():
        shutil.copy2(outputs_case["restore_txt"], pixeles_case)
    rstd_args = [str(pixeles_case), str(outputs_case["raster_T"]), str(outputs_case["mapa_T"]), str(outputs_case["raster_T_std"]), resol, str(outputs_case["mapa_std_T"])]
    rc, out = run_step(f"Raster STD T ({case_name})", scripts["raster_STD_T"], rstd_args, [outputs_case["mapa_T"], outputs_case["raster_T_std"], outputs_case["mapa_std_T"]], step_force("raster_std_t"))
    resumen.extend(out)
    if rc != 0:
        raise RuntimeError(f"[{case_name}] Raster STD T falló (code {rc})")
    resumen.append("")
    zero_args = [str(pixeles_case), str(outputs_case["raster_Z"]), resol]
    rc, out = run_step(f"Cálculo Termino Orden Cero ({case_name})", scripts["zero_zeta"], zero_args, [], step_force("zero_zeta"))
    resumen.extend(out)
    if rc != 0:
        raise RuntimeError(f"[{case_name}] Termino Orden Cero falló (code {rc})")
    resumen.append("")
    crear_reporte(resumen, outputs_case["report_case"])
    logger.info(f"Reporte del caso guardado en: {outputs_case['report_case']}")
    logger.info(f"=== Caso {case_name} finalizado ===")
    return resumen

# PARSEO Y EJECUCIÓN

def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pipeline robusto para generación, VCE, LC y casos", add_help=True)
    parser.add_argument("--freq", nargs="+", default=[DEFAULT_FREQ], dest="freqs")
    parser.add_argument("--resol", default=DEFAULT_RESOL, dest="resol")
    parser.add_argument("--lat-max", "--lat_max", default=DEFAULT_LAT_MAX, dest="lat_max")
    parser.add_argument("--lat-min", "--lat_min", default=DEFAULT_LAT_MIN, dest="lat_min")
    parser.add_argument("--lon-max", "--lon_max", default=DEFAULT_LON_MAX, dest="lon_max")
    parser.add_argument("--lon-min", "--lon_min", default=DEFAULT_LON_MIN, dest="lon_min")
    parser.add_argument("--apertura", "--Apertura", default=DEFAULT_APERTURA, dest="apertura")
    parser.add_argument("--si-no", "--Sino", default=DEFAULT_SINO, dest="si_no")
    parser.add_argument("--puntos", type=Path, default=DEFAULT_INPUTS["puntos"])
    parser.add_argument("--puntos-aero", "--puntos_aero", type=Path, default=DEFAULT_INPUTS["puntos_aero"], dest="puntos_aero")
    parser.add_argument("--listado", type=Path, default=DEFAULT_INPUTS["listado"])
    parser.add_argument("--sigma-file", "--sigma_file", type=Path, default=DEFAULT_SIGMA_FILE, dest="sigma_file")
    parser.add_argument("--obs-lat-col", default="Lat_Magna", dest="obs_lat_col")
    parser.add_argument("--obs-lon-col", default="Long_Magna", dest="obs_lon_col")
    parser.add_argument("--obs-alt-col", default="h", dest="obs_alt_col")
    parser.add_argument("--aero-lat-col", default="Latitud", dest="aero_lat_col")
    parser.add_argument("--aero-lon-col", default="Longitud", dest="aero_lon_col")
    parser.add_argument("--aero-alt-col", default="Altura", dest="aero_alt_col")
    parser.add_argument("--tam-bloque", "--tam_bloque", type=int, default=DEFAULT_BLOCK_SIZE, dest="tam_bloque")
    parser.add_argument("--obs-filter", default="Shannon", dest="obs_filter")
    parser.add_argument("--aero-filter", default="Cup", dest="aero_filter")
    parser.add_argument("--pixel-filter", default="Cup", dest="pixel_filter")
    parser.add_argument("--cases", nargs="+", choices=list(CASE_DEFINITIONS.keys()), default=list(DEFAULT_CASES), help="Vector de casos a calcular")
    parser.add_argument("--force-mode", choices=["none", "all", "steps"], default="none")
    parser.add_argument("--force-steps", nargs="+", default=[])
    parser.add_argument("--restore", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--qgeoid", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args(argv)

def build_config_from_args(args: argparse.Namespace) -> PipelineConfig:
    return PipelineConfig(
        freqs=[str(f) for f in args.freqs],
        resol=str(args.resol),
        lat_max=str(args.lat_max),
        lat_min=str(args.lat_min),
        lon_max=str(args.lon_max),
        lon_min=str(args.lon_min),
        apertura=str(args.apertura),
        si_no=str(args.si_no),

        puntos=resolve_input_path(args.puntos, "3_Observaciones"),
        puntos_aero=resolve_input_path(args.puntos_aero, "3_Observaciones"),
        listado=resolve_input_path(args.listado, "5_Main"),
        sigma_file=resolve_input_path(args.sigma_file, "3_Observaciones"),

        cases=list(args.cases),
        force_mode=str(args.force_mode),
        force_steps=[str(x) for x in args.force_steps],
        restore=normalize_bool(args.restore),
        qgeoid=normalize_bool(args.qgeoid),
        design_cols=DesignColumns(
            obs_lat=args.obs_lat_col,
            obs_lon=args.obs_lon_col,
            obs_alt=args.obs_alt_col,
            aero_lat=args.aero_lat_col,
            aero_lon=args.aero_lon_col,
            aero_alt=args.aero_alt_col,
            block_size=int(args.tam_bloque),
        ),
        design_filters=DesignFilters(
            obs_filter=args.obs_filter,
            aero_filter=args.aero_filter,
            pixel_filter=args.pixel_filter,
        ),
    )

def resolve_cases(selected_cases: Sequence[str]) -> List[Tuple[bool, bool, str]]:
    cases: List[Tuple[bool, bool, str]] = []
    for case_name in selected_cases:
        if case_name not in CASE_DEFINITIONS:
            raise ValueError(f"Caso inválido: {case_name}")
        use_ones, do_lc = CASE_DEFINITIONS[case_name]
        cases.append((use_ones, do_lc, case_name))
    return cases

def run_pipeline(config: PipelineConfig) -> None:
    input_paths = {"puntos": Path(config.puntos), "puntos_aero": Path(config.puntos_aero), "listado": Path(config.listado)}
    restore_pixel_outputs = build_outputs_restore_pixel()
    if config.qgeoid and not config.restore:
        logger.warning("QGeoid requiere Restore. Se activa Restore automáticamente.")
        config.restore = True
    freqs = [str(f) for f in config.freqs]
    cases = resolve_cases(config.cases)
    for freq in freqs:
        logger.info(f"--- frecuencia {freq} ---")
        hora_inicio = datetime.now()
        logger.info(f"Hora de inicio: {hora_inicio}")
        monitor = MonitorRecursos()
        monitor.start()
        reporte_global: List[str] = [f"======== REPORTE GLOBAL f{freq} ========\n"]
        resumen_global_pixel = run_restore_pixel(config.lat_max, config.lat_min, config.lon_max, config.lon_min, config.resol, restore_pixel_outputs, force=should_force_step(config, "restore_pixel"))
        reporte_global.append("===== PASO GLOBAL RESTORE_PIXEL =====")
        reporte_global.extend(resumen_global_pixel)
        reporte_global.append("")
        outputs_common, resumen_common = run_common(freq, config.resol, config.lat_max, config.lat_min, config.lon_max, config.lon_min, config.apertura, config.si_no, should_force_step(config, "common"), restore_pixel_outputs, input_paths, config.design_cols, config.design_filters)
        reporte_global.append("===== COMMON =====")
        reporte_global.extend(resumen_common)
        reporte_global.append("")
        sigma_types_needed = {use_ones for (use_ones, do_lc, name) in cases}
        shared_vce_outputs: Dict[Tuple[bool, bool], Dict[str, Path]] = {}
        vce_plan: Dict[bool, bool] = {}
        for use_ones, do_lc, _ in cases:
            if not do_lc:
                vce_plan[use_ones] = True
            else:
                tag = "ones" if use_ones else "sigmas"
                shared_dir = BASE_DIR / "6_Resultados" / f"f{freq}" / "COMMON" / f"VCE_shared_{tag}"
                pesos = shared_dir / f"pesos_{tag}_f{freq}.csv"
                normas = shared_dir / f"Normas_{tag}_f{freq}.txt"
                if pesos.exists() and normas.exists():
                    logger.info(f"[LC] Ya existen pesos/normas para {tag}, no se corre VCE")
                    continue
                if use_ones not in vce_plan:
                    vce_plan[use_ones] = False
        for use_ones, boolCov in vce_plan.items():
            step_key = f"vce_shared_{'ones' if use_ones else 'sigmas'}_{'cov' if boolCov else 'nocov'}"
            shared_vce_outputs[(use_ones, boolCov)], vce_log = run_shared_vce(freq, use_ones, config.sigma_file, input_paths, force=should_force_step(config, step_key), boolCov=boolCov)
            tag = "ones" if use_ones else "sigmas"
            cov_tag = "conCov" if boolCov else "sinCov"
            reporte_vce = BASE_DIR / "6_Resultados" / f"f{freq}" / "COMMON" / f"Reporte_VCE_{tag}_{cov_tag}_f{freq}.txt"
            crear_reporte(vce_log, reporte_vce)
            reporte_global.append(f"===== VCE ({tag}, {cov_tag}) =====")
            reporte_global.extend(vce_log)
            reporte_global.append("")
        shared_lc_outputs: Dict[bool, Dict[str, Path]] = {}
        if any(do_lc for (_, do_lc, _) in cases):
            for use_ones in sigma_types_needed:
                step_key = f"lcurve_shared_{'ones' if use_ones else 'sigmas'}"
                shared_lc_outputs[use_ones], lc_log = run_shared_lcurve(freq, use_ones, shared_vce_outputs, input_paths, force=should_force_step(config, step_key))
                reporte_lc = BASE_DIR / "6_Resultados" / f"f{freq}" / "COMMON" / f"Reporte_LC_{'ones' if use_ones else 'sigmas'}_f{freq}.txt"
                crear_reporte(lc_log, reporte_lc)
                tag = "ones" if use_ones else "sigmas"
                reporte_global.append(f"===== L-CURVE ({tag}) =====")
                reporte_global.extend(lc_log)
                reporte_global.append("")
        for use_ones, do_lc, case_name in cases:
            try:
                resumen_case = run_case(freq, case_name, use_ones, do_lc, outputs_common, restore_pixel_outputs, shared_vce_outputs, shared_lc_outputs, config.resol, config.force_mode, config.force_steps, input_paths, activate_restore=config.restore, activate_qgeoid=config.qgeoid)
                reporte_global.append("=" * 60)
                reporte_global.append(f"CASO: {case_name}")
                reporte_global.append("=" * 60)
                reporte_global.extend(resumen_case)
                reporte_global.append("")
            except Exception as e:
                logger.exception(f"Error en caso {case_name}: {e}")
        hora_fin = datetime.now()
        duracion = hora_fin - hora_inicio
        logger.info(f"Hora de fin: {hora_fin}")
        logger.info(f"Duración total: {duracion}")
        reporte_global.append("\n===== TIEMPO DE EJECUCIÓN =====")
        reporte_global.append(f"Hora de inicio: {hora_inicio}")
        reporte_global.append(f"Hora de fin: {hora_fin}")
        reporte_global.append(f"Duración total: {duracion}")
        reporte_global.append("")
        monitor.stop()
        base = R("6_Resultados", f"f{freq}", f"recursos_f{freq}")
        monitor.save_npy(base.with_suffix(".npy"))
        monitor.save_html(base.with_suffix(".html"))
        logger.info(f"Monitor guardado en: {base}")
        try:
            reporte_global.append("===== USO DE RECURSOS =====")
            reporte_global.extend(monitor.resumen())
            reporte_global.append("")
        except Exception:
            pass
        reporte_common = BASE_DIR / "6_Resultados" / f"f{freq}" / "COMMON" / f"Reporte_Common_f{freq}.txt"
        crear_reporte(resumen_common, reporte_common)
        logger.info(f"Reporte común guardado en: {reporte_common}")
        reporte_global_path = BASE_DIR / "6_Resultados" / f"f{freq}" / f"Reporte_GLOBAL_f{freq}.txt"
        crear_reporte(reporte_global, reporte_global_path)
        logger.info(f"Reporte GLOBAL guardado en: {reporte_global_path}")
    logger.info("FIN")

def main(argv=None):
    if argv is None:
        argv = sys.argv[1:]
    args = parse_args(argv)
    config = build_config_from_args(args)
    run_pipeline(config)

if __name__ == "__main__":
    main()
