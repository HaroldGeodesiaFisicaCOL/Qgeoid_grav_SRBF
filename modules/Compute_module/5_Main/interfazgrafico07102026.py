import io
import sys
import queue
import threading
import traceback
import contextlib
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

from Pipeline_07102026 import main


def build_args_from_form(values: dict) -> list[str]:
    args = []

    freq = [x.strip() for x in values["freq"].split(",") if x.strip()]
    if freq:
        args += ["--freq", *freq]

    args += ["--resol", values["resol"]]
    args += ["--lat-max", values["lat_max"]]
    args += ["--lat-min", values["lat_min"]]
    args += ["--lon-max", values["lon_max"]]
    args += ["--lon-min", values["lon_min"]]
    args += ["--apertura", values["apertura"]]
    args += ["--si-no", values["si_no"]]

    args += ["--puntos", values["puntos"]]
    args += ["--puntos-aero", values["puntos_aero"]]
    args += ["--listado", values["listado"]]
    args += ["--sigma-file", values["sigma_file"]]

    args += ["--obs-lat-col", values["obs_lat_col"]]
    args += ["--obs-lon-col", values["obs_lon_col"]]
    args += ["--obs-alt-col", values["obs_alt_col"]]

    args += ["--aero-lat-col", values["aero_lat_col"]]
    args += ["--aero-lon-col", values["aero_lon_col"]]
    args += ["--aero-alt-col", values["aero_alt_col"]]

    args += ["--pixel-lat-col", values["pixel_lat_col"]]
    args += ["--pixel-lon-col", values["pixel_lon_col"]]
    args += ["--pixel-alt-col", values["pixel_alt_col"]]

    args += ["--tam-bloque", values["tam_bloque"]]
    args += ["--kernel-block", values["kernel_block"]]

    args += ["--obs-filter", values["obs_filter"]]
    args += ["--aero-filter", values["aero_filter"]]
    args += ["--pixel-filter", values["pixel_filter"]]

    cases = values["cases"]
    if cases:
        args += ["--cases", *cases]

    args += ["--force-mode", values["force_mode"]]

    force_steps = [x.strip() for x in values["force_steps"].split(",") if x.strip()]
    if force_steps:
        args += ["--force-steps", *force_steps]

    if values["legacy"]:
        args.append("--legacy")
    if values["restore"]:
        args.append("--restore")
    if values["qgeoid"]:
        args.append("--qgeoid")

    return args


class QueueWriter(io.TextIOBase):
    def __init__(self, q: queue.Queue):
        super().__init__()
        self.q = q

    def write(self, s):
        if s:
            self.q.put(s)
        return len(s)

    def flush(self):
        pass


class PipelineGUI(tk.Tk):
    def __init__(self):
        super().__init__()

        self.title("Pipeline Geodésico - GUI")
        self.geometry("1180x900")
        self.minsize(1080, 780)

        self.log_queue = queue.Queue()
        self.worker_thread = None

        self._build_vars()
        self._build_ui()
        self.after(100, self._poll_log_queue)

    def _build_vars(self):
        self.vars = {
            "freq": tk.StringVar(value="606,615"),
            "resol": tk.StringVar(value="1"),

            "lat_max": tk.StringVar(value="5.890357850333333"),
            "lat_min": tk.StringVar(value="1.8903578503333334"),
            "lon_max": tk.StringVar(value="-75.01875315333334"),
            "lon_min": tk.StringVar(value="-79.01875315333334"),
            "apertura": tk.StringVar(value="0.4"),

            "si_no": tk.StringVar(value="Si"),

            "puntos": tk.StringVar(value="Obs_Terr_DTU21Gra_Nosotros.txt"),
            "puntos_aero": tk.StringVar(value="Datos_Aereos_Sesg_900.txt"),
            "listado": tk.StringVar(value="listado.txt"),
            "sigma_file": tk.StringVar(value="Informe_varianza_x_proyectos_f2190.txt"),

            "obs_lat_col": tk.StringVar(value="Lat_Magna"),
            "obs_lon_col": tk.StringVar(value="Long_Magna"),
            "obs_alt_col": tk.StringVar(value="h"),

            "aero_lat_col": tk.StringVar(value="latitud"),
            "aero_lon_col": tk.StringVar(value="longitud"),
            "aero_alt_col": tk.StringVar(value="h"),

            "pixel_lat_col": tk.StringVar(value="Latitud"),
            "pixel_lon_col": tk.StringVar(value="Longitud"),
            "pixel_alt_col": tk.StringVar(value="altura"),

            "tam_bloque": tk.StringVar(value="5000"),
            "kernel_block": tk.StringVar(value="2000"),

            "obs_filter": tk.StringVar(value="Shannon"),
            "aero_filter": tk.StringVar(value="Cup"),
            "pixel_filter": tk.StringVar(value="Cup"),

            "force_mode": tk.StringVar(value="none"),
            "force_steps": tk.StringVar(value=""),

            "legacy": tk.BooleanVar(value=False),
            "restore": tk.BooleanVar(value=True),
            "qgeoid": tk.BooleanVar(value=True),

            "case_1VC": tk.BooleanVar(value=True),
            "case_1VCLC": tk.BooleanVar(value=True),
            "case_SigmasVC": tk.BooleanVar(value=False),
            "case_SigmasVCLC": tk.BooleanVar(value=False),
        }

    def _build_ui(self):
        self.columnconfigure(0, weight=1)
        self.rowconfigure(1, weight=1)

        top = ttk.Frame(self, padding=10)
        top.grid(row=0, column=0, sticky="ew")
        top.columnconfigure(0, weight=1)

        btns = ttk.Frame(top)
        btns.grid(row=0, column=0, sticky="ew")
        btns.columnconfigure(10, weight=1)

        ttk.Button(btns, text="Ejecutar", command=self.run_pipeline).grid(row=0, column=0, padx=5)
        ttk.Button(btns, text="Limpiar consola", command=self.clear_log).grid(row=0, column=1, padx=5)
        ttk.Button(btns, text="Salir", command=self.destroy).grid(row=0, column=2, padx=5)

        self.progress = ttk.Progressbar(btns, mode="indeterminate")
        self.progress.grid(row=0, column=10, sticky="ew", padx=10)

        main_frame = ttk.PanedWindow(self, orient="vertical")
        main_frame.grid(row=1, column=0, sticky="nsew")

        form_container = ttk.Notebook(main_frame)
        main_frame.add(form_container, weight=4)

        log_container = ttk.Frame(main_frame, padding=6)
        main_frame.add(log_container, weight=2)

        self._build_tabs(form_container)
        self._build_log(log_container)

    def _build_tabs(self, nb: ttk.Notebook):
        tab_general = ttk.Frame(nb, padding=12)
        tab_files = ttk.Frame(nb, padding=12)
        tab_columns = ttk.Frame(nb, padding=12)
        tab_process = ttk.Frame(nb, padding=12)

        nb.add(tab_general, text="General")
        nb.add(tab_files, text="Archivos")
        nb.add(tab_columns, text="Columnas")
        nb.add(tab_process, text="Proceso")

        self._build_general_tab(tab_general)
        self._build_files_tab(tab_files)
        self._build_columns_tab(tab_columns)
        self._build_process_tab(tab_process)

    def _build_general_tab(self, parent):
        for i in range(4):
            parent.columnconfigure(i, weight=1)

        fields = [
            ("Frecuencias (coma)", "freq"),
            ("Resolución", "resol"),
            ("Lat Max", "lat_max"),
            ("Lat Min", "lat_min"),
            ("Lon Max", "lon_max"),
            ("Lon Min", "lon_min"),
            ("Apertura", "apertura"),
            ("Si/No", "si_no"),
            ("Tamaño bloque (diseño)", "tam_bloque"),
            ("Kernel block (in-situ)", "kernel_block"),
            ("Obs Filter", "obs_filter"),
            ("Aero Filter", "aero_filter"),
            ("Pixel Filter", "pixel_filter"),
        ]

        for idx, (label, key) in enumerate(fields):
            r = idx // 2
            c = (idx % 2) * 2
            ttk.Label(parent, text=label).grid(row=r, column=c, sticky="w", padx=5, pady=5)
            ttk.Entry(parent, textvariable=self.vars[key]).grid(row=r, column=c + 1, sticky="ew", padx=5, pady=5)

    def _build_files_tab(self, parent):
        parent.columnconfigure(1, weight=1)

        file_fields = [
            ("Puntos", "puntos"),
            ("Puntos Aero", "puntos_aero"),
            ("Listado", "listado"),
            ("Sigma File", "sigma_file"),
        ]

        for i, (label, key) in enumerate(file_fields):
            ttk.Label(parent, text=label).grid(row=i, column=0, sticky="w", padx=5, pady=8)
            ttk.Entry(parent, textvariable=self.vars[key]).grid(row=i, column=1, sticky="ew", padx=5, pady=8)
            ttk.Button(parent, text="Examinar...", command=lambda k=key: self.pick_file(k)).grid(row=i, column=2, padx=5, pady=8)

    def _build_columns_tab(self, parent):
        for i in range(6):
            parent.columnconfigure(i, weight=1)

        ttk.Label(parent, text="Observaciones", font=("Segoe UI", 10, "bold")).grid(row=0, column=0, columnspan=2, sticky="w", pady=10)
        ttk.Label(parent, text="Aero", font=("Segoe UI", 10, "bold")).grid(row=0, column=2, columnspan=2, sticky="w", pady=10)
        ttk.Label(parent, text="Píxeles", font=("Segoe UI", 10, "bold")).grid(row=0, column=4, columnspan=2, sticky="w", pady=10)

        obs = [("Lat", "obs_lat_col"), ("Lon", "obs_lon_col"), ("Alt", "obs_alt_col")]
        aero = [("Lat", "aero_lat_col"), ("Lon", "aero_lon_col"), ("Alt", "aero_alt_col")]
        pix = [("Lat", "pixel_lat_col"), ("Lon", "pixel_lon_col"), ("Alt", "pixel_alt_col")]

        for i, (lab, key) in enumerate(obs, start=1):
            ttk.Label(parent, text=lab).grid(row=i, column=0, sticky="w", padx=5, pady=5)
            ttk.Entry(parent, textvariable=self.vars[key]).grid(row=i, column=1, sticky="ew", padx=5, pady=5)

        for i, (lab, key) in enumerate(aero, start=1):
            ttk.Label(parent, text=lab).grid(row=i, column=2, sticky="w", padx=5, pady=5)
            ttk.Entry(parent, textvariable=self.vars[key]).grid(row=i, column=3, sticky="ew", padx=5, pady=5)

        for i, (lab, key) in enumerate(pix, start=1):
            ttk.Label(parent, text=lab).grid(row=i, column=4, sticky="w", padx=5, pady=5)
            ttk.Entry(parent, textvariable=self.vars[key]).grid(row=i, column=5, sticky="ew", padx=5, pady=5)

    def _build_process_tab(self, parent):
        parent.columnconfigure(0, weight=1)

        box_cases = ttk.LabelFrame(parent, text="Casos", padding=10)
        box_cases.grid(row=0, column=0, sticky="ew", padx=5, pady=5)

        ttk.Checkbutton(box_cases, text="1VC", variable=self.vars["case_1VC"]).grid(row=0, column=0, padx=8, pady=5, sticky="w")
        ttk.Checkbutton(box_cases, text="1VCLC", variable=self.vars["case_1VCLC"]).grid(row=0, column=1, padx=8, pady=5, sticky="w")
        ttk.Checkbutton(box_cases, text="SigmasVC", variable=self.vars["case_SigmasVC"]).grid(row=1, column=0, padx=8, pady=5, sticky="w")
        ttk.Checkbutton(box_cases, text="SigmasVCLC", variable=self.vars["case_SigmasVCLC"]).grid(row=1, column=1, padx=8, pady=5, sticky="w")

        box_proc = ttk.LabelFrame(parent, text="Procesos", padding=10)
        box_proc.grid(row=1, column=0, sticky="ew", padx=5, pady=10)

        ttk.Checkbutton(box_proc, text="Restore", variable=self.vars["restore"]).grid(row=0, column=0, padx=8, pady=5, sticky="w")
        ttk.Checkbutton(box_proc, text="QGeoid", variable=self.vars["qgeoid"]).grid(row=0, column=1, padx=8, pady=5, sticky="w")

        box_mode = ttk.LabelFrame(parent, text="Modo de cálculo", padding=10)
        box_mode.grid(row=2, column=0, sticky="ew", padx=5, pady=10)

        ttk.Checkbutton(
            box_mode,
            text="Legacy (matriz de diseño píxeles en disco)",
            variable=self.vars["legacy"],
        ).grid(row=0, column=0, columnspan=2, padx=8, pady=5, sticky="w")

        ttk.Label(
            box_mode,
            text="Sin marcar → in-situ (sin matriz .npy, cálculo en GPU al vuelo).",
        ).grid(row=1, column=0, columnspan=2, padx=8, pady=2, sticky="w")

        box_force = ttk.LabelFrame(parent, text="Force", padding=10)
        box_force.grid(row=3, column=0, sticky="ew", padx=5, pady=10)
        box_force.columnconfigure(1, weight=1)

        ttk.Label(box_force, text="Force Mode").grid(row=0, column=0, sticky="w", padx=5, pady=5)
        ttk.Combobox(
            box_force,
            textvariable=self.vars["force_mode"],
            values=["none", "all", "steps"],
            state="readonly",
            width=12,
        ).grid(row=0, column=1, sticky="w", padx=5, pady=5)

        ttk.Label(box_force, text="Force Steps (coma)").grid(row=1, column=0, sticky="w", padx=5, pady=5)
        ttk.Entry(box_force, textvariable=self.vars["force_steps"]).grid(row=1, column=1, sticky="ew", padx=5, pady=5)

        ttk.Label(
            box_force,
            text="Ejemplos de steps: restore_pixel, common, diseno_pixeles, "
                 "raster_t, restore, std, qgeoid, visu, significancia, "
                 "raster_std_t, zero_zeta, vce_shared_ones_cov, "
                 "lcurve_shared_ones.",
            wraplength=980,
            justify="left",
        ).grid(row=2, column=0, columnspan=2, sticky="w", padx=5, pady=5)

    def _build_log(self, parent):
        parent.rowconfigure(0, weight=1)
        parent.columnconfigure(0, weight=1)

        self.log_text = tk.Text(parent, wrap="word", height=18)
        self.log_text.grid(row=0, column=0, sticky="nsew")

        scroll = ttk.Scrollbar(parent, orient="vertical", command=self.log_text.yview)
        scroll.grid(row=0, column=1, sticky="ns")
        self.log_text.configure(yscrollcommand=scroll.set)

    def log(self, text: str):
        self.log_text.insert("end", text)
        self.log_text.see("end")

    def clear_log(self):
        self.log_text.delete("1.0", "end")

    def _poll_log_queue(self):
        try:
            while True:
                msg = self.log_queue.get_nowait()
                self.log(msg)
        except queue.Empty:
            pass
        self.after(100, self._poll_log_queue)

    def pick_file(self, key):
        path = filedialog.askopenfilename()
        if path:
            self.vars[key].set(path)

    def _collect_values(self):
        cases = []
        if self.vars["case_1VC"].get():
            cases.append("1VC")
        if self.vars["case_1VCLC"].get():
            cases.append("1VCLC")
        if self.vars["case_SigmasVC"].get():
            cases.append("SigmasVC")
        if self.vars["case_SigmasVCLC"].get():
            cases.append("SigmasVCLC")

        values = {
            "freq": self.vars["freq"].get(),
            "resol": self.vars["resol"].get(),
            "lat_max": self.vars["lat_max"].get(),
            "lat_min": self.vars["lat_min"].get(),
            "lon_max": self.vars["lon_max"].get(),
            "lon_min": self.vars["lon_min"].get(),
            "apertura": self.vars["apertura"].get(),
            "si_no": self.vars["si_no"].get(),

            "puntos": self.vars["puntos"].get(),
            "puntos_aero": self.vars["puntos_aero"].get(),
            "listado": self.vars["listado"].get(),
            "sigma_file": self.vars["sigma_file"].get(),

            "obs_lat_col": self.vars["obs_lat_col"].get(),
            "obs_lon_col": self.vars["obs_lon_col"].get(),
            "obs_alt_col": self.vars["obs_alt_col"].get(),

            "aero_lat_col": self.vars["aero_lat_col"].get(),
            "aero_lon_col": self.vars["aero_lon_col"].get(),
            "aero_alt_col": self.vars["aero_alt_col"].get(),

            "pixel_lat_col": self.vars["pixel_lat_col"].get(),
            "pixel_lon_col": self.vars["pixel_lon_col"].get(),
            "pixel_alt_col": self.vars["pixel_alt_col"].get(),

            "tam_bloque": self.vars["tam_bloque"].get(),
            "kernel_block": self.vars["kernel_block"].get(),

            "obs_filter": self.vars["obs_filter"].get(),
            "aero_filter": self.vars["aero_filter"].get(),
            "pixel_filter": self.vars["pixel_filter"].get(),

            "cases": cases,
            "force_mode": self.vars["force_mode"].get(),
            "force_steps": self.vars["force_steps"].get(),

            "legacy": self.vars["legacy"].get(),
            "restore": self.vars["restore"].get(),
            "qgeoid": self.vars["qgeoid"].get(),
        }
        return values

    def run_pipeline(self):
        if self.worker_thread and self.worker_thread.is_alive():
            messagebox.showwarning("En ejecución", "Ya hay un proceso ejecutándose.")
            return

        values = self._collect_values()

        if not values["cases"]:
            messagebox.showwarning("Casos", "Debe seleccionar al menos un caso.")
            return

        args = build_args_from_form(values)

        self.log("\n" + "=" * 80 + "\n")
        self.log("Ejecutando pipeline...\n")
        self.log(f"Modo legacy: {values['legacy']}\n")
        self.log("Argumentos:\n")
        self.log(" ".join(args) + "\n\n")

        self.progress.start(10)

        self.worker_thread = threading.Thread(
            target=self._worker_run,
            args=(args,),
            daemon=True,
        )
        self.worker_thread.start()

    def _worker_run(self, args):
        writer = QueueWriter(self.log_queue)

        try:
            with contextlib.redirect_stdout(writer), contextlib.redirect_stderr(writer):
                main(args)

            self.log_queue.put("\nProceso finalizado correctamente.\n")

        except Exception:
            self.log_queue.put("\n[ERROR] Falló la ejecución:\n")
            self.log_queue.put(traceback.format_exc())

        finally:
            self.progress.stop()


if __name__ == "__main__":
    app = PipelineGUI()
    app.mainloop()