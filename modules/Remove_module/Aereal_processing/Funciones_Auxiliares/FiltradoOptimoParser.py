# -*- coding: utf-8 -*-
from __future__ import annotations

import math
import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from numba import cuda, njit, prange

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(x, **kwargs):
        return x


@dataclass(frozen=True)
class FilterConfig:
    input_file: Path
    output_dir: Path

    project_column: str = "Proyecto"
    line_column: str = "Num_Linea"
    lat_column: str = "Latitud"
    lon_column: str = "Longitud"
    input_column: str = "Pre_Pert_Grav"
    output_column: str = "Pre_Pert_Grav_Filt"

    hwhm_km: float = 3.0
    earth_radius_km: float = 6378.137
    use_gpu: bool = True
    gpu_threshold: int = 8000
    cpu_parallel_threshold: int = 800


def safe_filename(text) -> str:
    text = str(text)
    for ch in ['/', '\\', ':', '*', '?', '"', '<', '>', '|']:
        text = text.replace(ch, "_")
    return text.replace(" ", "_").strip("_")


def read_input_file(path: str | Path, cfg: FilterConfig) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"No existe el archivo: {p}")

    df = pd.read_csv(p, sep="\t", encoding="utf-8-sig", low_memory=False)

    for col in [cfg.lat_column, cfg.lon_column, cfg.line_column, cfg.input_column]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def save_output(original_path: str | Path, output_dir: Path, df_out: pd.DataFrame, stats: pd.DataFrame) -> tuple[Path, Path]:
    original_path = Path(original_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    out_path = output_dir / f"{original_path.stem}_filtrado.txt"
    stats_path = output_dir / f"{original_path.stem}_resumen_lineas.txt"

    df_out.to_csv(out_path, sep="\t", index=False, encoding="utf-8-sig")
    stats.to_csv(stats_path, sep="\t", index=False, encoding="utf-8-sig")

    return out_path, stats_path


def gaussian_b(hwhm_km: float, earth_radius_km: float) -> float:
    sigma_km = hwhm_km / math.sqrt(2.0 * math.log(2.0))
    return (earth_radius_km ** 2) / (sigma_km ** 2)


@njit(cache=True)
def gaussian_filter_cpu_serial(lat_rad, lon_rad, values, sin_lat, cos_lat, b):
    m = values.shape[0]
    out = np.empty(m, dtype=np.float64)

    for i in range(m):
        si = sin_lat[i]
        ci = cos_lat[i]

        sum_w = 0.0
        sum_y = 0.0

        for j in range(m):
            dlon = lon_rad[j] - lon_rad[i]
            cos_psi = si * sin_lat[j] + ci * cos_lat[j] * math.cos(dlon)

            if cos_psi > 1.0:
                cos_psi = 1.0
            elif cos_psi < -1.0:
                cos_psi = -1.0

            w = math.exp(-b * (1.0 - cos_psi))
            sum_w += w
            sum_y += w * values[j]

        out[i] = sum_y / sum_w if sum_w > 0.0 else np.nan

    return out


@njit(parallel=True, cache=True)
def gaussian_filter_cpu_parallel(lat_rad, lon_rad, values, sin_lat, cos_lat, b):
    m = values.shape[0]
    out = np.empty(m, dtype=np.float64)

    for i in prange(m):
        si = sin_lat[i]
        ci = cos_lat[i]

        sum_w = 0.0
        sum_y = 0.0

        for j in range(m):
            dlon = lon_rad[j] - lon_rad[i]
            cos_psi = si * sin_lat[j] + ci * cos_lat[j] * math.cos(dlon)

            if cos_psi > 1.0:
                cos_psi = 1.0
            elif cos_psi < -1.0:
                cos_psi = -1.0

            w = math.exp(-b * (1.0 - cos_psi))
            sum_w += w
            sum_y += w * values[j]

        out[i] = sum_y / sum_w if sum_w > 0.0 else np.nan

    return out


@cuda.jit
def gaussian_filter_cuda(lat_rad, lon_rad, values, sin_lat, cos_lat, out, b):
    i = cuda.grid(1)
    m = values.shape[0]

    if i >= m:
        return

    si = sin_lat[i]
    ci = cos_lat[i]

    sum_w = 0.0
    sum_y = 0.0

    for j in range(m):
        dlon = lon_rad[j] - lon_rad[i]
        cos_psi = si * sin_lat[j] + ci * cos_lat[j] * math.cos(dlon)

        if cos_psi > 1.0:
            cos_psi = 1.0
        elif cos_psi < -1.0:
            cos_psi = -1.0

        w = math.exp(-b * (1.0 - cos_psi))
        sum_w += w
        sum_y += w * values[j]

    out[i] = sum_y / sum_w if sum_w > 0.0 else np.nan


def filter_line_cpu(lat_deg, lon_deg, values, b_gauss: float, cpu_parallel_threshold: int):
    lat_deg = np.asarray(lat_deg, dtype=np.float64)
    lon_deg = np.asarray(lon_deg, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)

    out_full = np.full(values.shape[0], np.nan, dtype=np.float64)

    valid = np.isfinite(lat_deg) & np.isfinite(lon_deg) & np.isfinite(values)
    if not np.any(valid):
        return out_full

    lat_rad = np.radians(lat_deg[valid])
    lon_rad = np.radians(lon_deg[valid])
    vals = values[valid]

    sin_lat = np.sin(lat_rad)
    cos_lat = np.cos(lat_rad)

    if vals.size >= cpu_parallel_threshold:
        filtered_valid = gaussian_filter_cpu_parallel(lat_rad, lon_rad, vals, sin_lat, cos_lat, b_gauss)
    else:
        filtered_valid = gaussian_filter_cpu_serial(lat_rad, lon_rad, vals, sin_lat, cos_lat, b_gauss)

    out_full[valid] = filtered_valid
    return out_full


def filter_line_gpu(lat_deg, lon_deg, values, b_gauss: float):
    lat_deg = np.asarray(lat_deg, dtype=np.float64)
    lon_deg = np.asarray(lon_deg, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)

    out_full = np.full(values.shape[0], np.nan, dtype=np.float64)

    valid = np.isfinite(lat_deg) & np.isfinite(lon_deg) & np.isfinite(values)
    if not np.any(valid):
        return out_full

    lat_rad = np.radians(lat_deg[valid])
    lon_rad = np.radians(lon_deg[valid])
    vals = values[valid]

    sin_lat = np.sin(lat_rad)
    cos_lat = np.cos(lat_rad)

    d_lon = cuda.to_device(lon_rad)
    d_val = cuda.to_device(vals)
    d_sin = cuda.to_device(sin_lat)
    d_cos = cuda.to_device(cos_lat)
    d_out = cuda.device_array(vals.shape[0], dtype=np.float64)

    threads = 256
    blocks = (vals.shape[0] + threads - 1) // threads

    gaussian_filter_cuda[blocks, threads](d_lon, d_lon, d_val, d_sin, d_cos, d_out, b_gauss)
    filtered_valid = d_out.copy_to_host()

    out_full[valid] = filtered_valid
    return out_full


def gaussian_filter_line(lat_deg, lon_deg, values, cfg: FilterConfig, b_gauss: float):
    if cfg.use_gpu and cuda.is_available() and len(values) >= cfg.gpu_threshold:
        return filter_line_gpu(lat_deg, lon_deg, values, b_gauss)
    return filter_line_cpu(lat_deg, lon_deg, values, b_gauss, cfg.cpu_parallel_threshold)


def line_stats(cfg: FilterConfig, project, line, values, filtered):
    diff = values - filtered
    abs_diff = np.abs(diff)

    finite_abs = np.isfinite(abs_diff)
    finite_diff = np.isfinite(diff)

    return {
        cfg.project_column: project,
        cfg.line_column: line,
        "n_puntos": int(len(values)),
        "nan_entrada": int(np.isnan(values).sum()),
        "nan_salida": int(np.isnan(filtered).sum()),
        "media_abs_diff": float(np.nanmean(abs_diff)) if finite_abs.any() else np.nan,
        "std_diff": float(np.nanstd(diff)) if finite_diff.any() else np.nan,
        "max_abs_diff": float(np.nanmax(abs_diff)) if finite_abs.any() else np.nan,
    }


def build_tasks(df: pd.DataFrame, cfg: FilterConfig):
    tasks = []
    for (project, line), group in df.groupby([cfg.project_column, cfg.line_column], sort=False):
        tasks.append({
            "project": project,
            "line": line,
            "index": group.index.to_numpy(),
            "lat": group[cfg.lat_column].to_numpy(dtype=np.float64),
            "lon": group[cfg.lon_column].to_numpy(dtype=np.float64),
            "values": group[cfg.input_column].to_numpy(dtype=np.float64),
            "n": int(len(group)),
        })
    return tasks


def finalize_result(df_out: pd.DataFrame, task, filtered, stats_rows, cfg: FilterConfig):
    df_out.loc[task["index"], cfg.output_column] = filtered
    stats_rows.append(line_stats(cfg, task["project"], task["line"], task["values"], filtered))


def process_dataframe(df: pd.DataFrame, cfg: FilterConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    required = [cfg.project_column, cfg.line_column, cfg.lat_column, cfg.lon_column, cfg.input_column]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"Faltan estas columnas obligatorias: {missing}")

    out = df.copy()
    out[cfg.output_column] = np.nan

    b_gauss = gaussian_b(cfg.hwhm_km, cfg.earth_radius_km)
    tasks = build_tasks(out, cfg)

    print(f"Proyectos detectados: {out[cfg.project_column].dropna().nunique()}")
    print(f"Líneas detectadas: {len(tasks)}")

    stats_all = []

    for task in tqdm(tasks, desc="Filtrando líneas", unit="línea"):
        filtered = gaussian_filter_line(task["lat"], task["lon"], task["values"], cfg, b_gauss)
        finalize_result(out, task, filtered, stats_all, cfg)

    stats = pd.DataFrame(stats_all)
    return out, stats


def make_text_summary(stats: pd.DataFrame, report_txt: Path, cfg: FilterConfig) -> pd.DataFrame:
    project_summary = (
        stats.groupby(cfg.project_column, dropna=False)
        .size()
        .reset_index(name="lineas_filtradas")
        .sort_values(["lineas_filtradas", cfg.project_column], ascending=[False, True])
    )

    project_summary[cfg.project_column] = project_summary[cfg.project_column].fillna("(sin proyecto)").astype(str)
    total_lines = int(len(stats))

    lines = []
    lines.append("REPORTE DE FILTRADO")
    lines.append("=" * 80)
    lines.append("Filtro usado: Filtro gaussiano esférico completo")
    lines.append("Función de peso: w_ij = exp(-b(1 - cos(ψ_ij)))")
    lines.append("b = R^2 / σ^2,  σ = HWHM / sqrt(2 ln 2)")
    lines.append(f"HWHM = {cfg.hwhm_km} km")
    lines.append(f"Radio terrestre = {cfg.earth_radius_km} km")
    lines.append("")
    lines.append(f"Total de líneas filtradas: {total_lines}")
    lines.append(f"Total de proyectos filtrados: {len(project_summary)}")
    lines.append("")
    lines.append("Líneas filtradas por proyecto:")
    for _, row in project_summary.iterrows():
        lines.append(f"  - {row[cfg.project_column]}: {int(row['lineas_filtradas'])}")

    report_txt.write_text("\n".join(lines), encoding="utf-8")
    return project_summary


def add_summary_page(pdf: PdfPages, project_summary: pd.DataFrame, total_lines: int, cfg: FilterConfig):
    fig = plt.figure(figsize=(11.69, 8.27))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")

    title = "Reporte único de filtrado"
    subtitle = "Filtro gaussiano esférico completo, sin ventana ni truncamiento matemático"

    txt = [
        "Filtro usado: Filtro gaussiano esférico completo",
        "Función de peso: w_ij = exp(-b(1 - cos(ψ_ij)))",
        "b = R^2 / σ^2,  σ = HWHM / sqrt(2 ln 2)",
        f"HWHM = {cfg.hwhm_km} km",
        f"Radio terrestre = {cfg.earth_radius_km} km",
        "",
        f"Total de líneas filtradas: {total_lines}",
        f"Total de proyectos filtrados: {len(project_summary)}",
    ]

    ax.text(0.5, 0.95, title, ha="center", va="top", fontsize=18, fontweight="bold")
    ax.text(0.5, 0.90, subtitle, ha="center", va="top", fontsize=11)
    ax.text(0.06, 0.82, "\n".join(txt), ha="left", va="top", fontsize=11)

    fig.tight_layout()
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def add_project_table_pages(pdf: PdfPages, project_summary: pd.DataFrame, cfg: FilterConfig, rows_per_page: int = 22):
    if project_summary.empty:
        fig = plt.figure(figsize=(11.69, 8.27))
        ax = fig.add_axes([0, 0, 1, 1])
        ax.axis("off")
        ax.text(0.5, 0.5, "No hay proyectos para mostrar.", ha="center", va="center", fontsize=12)
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)
        return

    df_show = project_summary.copy()
    df_show.columns = ["Proyecto", "Líneas filtradas"]

    n = len(df_show)
    for start in range(0, n, rows_per_page):
        chunk = df_show.iloc[start:start + rows_per_page]

        fig, ax = plt.subplots(figsize=(11.69, 8.27))
        ax.axis("off")

        page_no = start // rows_per_page + 1
        total_pages = (n + rows_per_page - 1) // rows_per_page

        ax.set_title(
            f"Proyectos filtrados y cantidad de líneas por proyecto (página {page_no} de {total_pages})",
            fontsize=13,
            pad=16
        )

        tbl = ax.table(
            cellText=chunk.values.tolist(),
            colLabels=chunk.columns.tolist(),
            cellLoc="center",
            loc="center"
        )
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(10)
        tbl.scale(1, 1.35)

        fig.tight_layout()
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)


def generate_unique_report(stats: pd.DataFrame, cfg: FilterConfig):
    report_pdf = cfg.output_dir / "reporte_filtrado.pdf"
    report_txt = cfg.output_dir / "reporte_filtrado.txt"

    project_summary = make_text_summary(stats, report_txt, cfg)
    total_lines = int(len(stats))

    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    with PdfPages(report_pdf) as pdf:
        add_summary_page(pdf, project_summary, total_lines, cfg)
        add_project_table_pages(pdf, project_summary, cfg, rows_per_page=22)

    return report_pdf, report_txt


def GaussianFilter(
    input_file: str | Path,
    output_dir: str | Path | None = None,
    project_column: str = "Proyecto",
    line_column: str = "Num_Linea",
    lat_column: str = "Latitud",
    lon_column: str = "Longitud",
    input_column: str = "Pre_Pert_Grav",
    output_column: str = "Pre_Pert_Grav_Filt",
    hwhm_km: float = 3.0,
    earth_radius_km: float = 6378.137,
    use_gpu: bool = True,
    gpu_threshold: int = 8000,
    cpu_parallel_threshold: int = 800,
):
    input_file = Path(input_file)

    if output_dir is None:
        output_dir = input_file.parent / "salida_filtrado"
    output_dir = Path(output_dir)

    cfg = FilterConfig(
        input_file=input_file,
        output_dir=output_dir,
        project_column=project_column,
        line_column=line_column,
        lat_column=lat_column,
        lon_column=lon_column,
        input_column=input_column,
        output_column=output_column,
        hwhm_km=hwhm_km,
        earth_radius_km=earth_radius_km,
        use_gpu=use_gpu,
        gpu_threshold=gpu_threshold,
        cpu_parallel_threshold=cpu_parallel_threshold,
    )

    df = read_input_file(cfg.input_file, cfg)
    df_out, stats = process_dataframe(df, cfg)
    out_path, stats_path = save_output(cfg.input_file, cfg.output_dir, df_out, stats)
    report_pdf, report_txt = generate_unique_report(stats, cfg)

    print("\n=== RESUMEN ===")
    print(f"Archivo filtrado: {out_path}")
    print(f"Archivo resumen de líneas: {stats_path}")
    print(f"Reporte único PDF: {report_pdf}")
    print(f"Reporte único TXT: {report_txt}")

    print("\nPrimeras filas del resumen:")
    if not stats.empty:
        print(stats.head().to_string(index=False))
    else:
        print("Sin estadísticas.")

    return df_out,report_txt


# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description="Filtrado gaussiano esférico de aerogravimetría por líneas.")
#     parser.add_argument("input_file", help="Archivo de entrada .txt o .csv con separador tabulado")
#     parser.add_argument("--output-dir", default=None, help="Carpeta de salida")

#     parser.add_argument("--project-column", default="Proyecto")
#     parser.add_argument("--line-column", default="Num_Linea")
#     parser.add_argument("--lat-column", default="Latitud")
#     parser.add_argument("--lon-column", default="Longitud")
#     parser.add_argument("--input-column", default="Pre_Pert_Grav")
#     parser.add_argument("--output-column", default="Pre_Pert_Grav_Filt")

#     parser.add_argument("--hwhm-km", type=float, default=3.0, help="HWHM en km")
#     parser.add_argument("--earth-radius-km", type=float, default=6378.137, help="Radio terrestre en km")
#     parser.add_argument("--no-gpu", action="store_true", help="Desactiva GPU")
#     parser.add_argument("--gpu-threshold", type=int, default=8000, help="Tamaño mínimo para usar GPU")
#     parser.add_argument("--cpu-parallel-threshold", type=int, default=800, help="Tamaño mínimo para prange")

#     args = parser.parse_args()

#     main(
#         input_file=args.input_file,
#         output_dir=args.output_dir,
#         project_column=args.project_column,
#         line_column=args.line_column,
#         lat_column=args.lat_column,
#         lon_column=args.lon_column,
#         input_column=args.input_column,
#         output_column=args.output_column,
#         hwhm_km=args.hwhm_km,
#         earth_radius_km=args.earth_radius_km,
#         use_gpu=not args.no_gpu,
#         gpu_threshold=args.gpu_threshold,
#         cpu_parallel_threshold=args.cpu_parallel_threshold,
#     )
