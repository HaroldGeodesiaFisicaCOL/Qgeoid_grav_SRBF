# gui_app.py
# Interfaz gráfica para Pipeline_05052026_argpase.py
#
# Requiere que exista:
#   from Pipeline_05052026_argpase import main
#
# Ejecuta:
#   python gui_app.py

import io
import sys
import queue
import threading
import traceback
import contextlib
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

from Pipeline_05052026_argpase import main


# ============================================================
# Construcción de argumentos
# ============================================================

def build_args_from_form(values: dict) -> list[str]:
    args = []

    # Frecuencias
    freq = [x.strip() for x in values["freq"].split(",") if x.strip()]
    if freq:
        args += ["--freq", *freq]

    # Escalares
    args += ["--resol", values["resol"]]
    args += ["--lat-max", values["lat_max"]]
    args += ["--lat-min", values["lat_min"]]
    args += ["--lon-max", values["lon_max"]]
    args += ["--lon-min", values["lon_min"]]
    args += ["--apertura", values["apertura"]]
    args += ["--si-no", values["si_no"]]

    # Archivos
    args += ["--puntos", values["puntos"]]
    args += ["--puntos-aero", values["puntos_aero"]]
    args += ["--listado", values["listado"]]
    args += ["--sigma-file", values["sigma_file"]]

    # Columnas OBS
    args += ["--obs-lat-col", values["obs_lat_col"]]
    args += ["--obs-lon-col", values["obs_lon_col"]]
    args += ["--obs-alt-col", values["obs_alt_col"]]

    # Columnas AERO
    args += ["--aero-lat-col", values["aero_lat_col"]]
    args += ["--aero-lon-col", values["aero_lon_col"]]
    args += ["--aero-alt-col", values["aero_alt_col"]]

    # Otros
    args += ["--tam-bloque", values["tam_bloque"]]
    args += ["--obs-filter", values["obs_filter"]]
    args += ["--aero-filter", values["aero_filter"]]
    args += ["--pixel-filter", values["pixel_filter"]]

    # Casos
    cases = values["cases"]
    if cases:
        args += ["--cases", *cases]

    args += ["--force-mode", values["force_mode"]]

    # Flags
    if values["restore"]:
        args.append("--restore")
    if values["qgeoid"]:
        args.append("--qgeoid")

    return args


# ============================================================
# Redirección stdout/stderr -> Cola GUI
# ============================================================

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


# ============================================================
# GUI
# ============================================================

class PipelineGUI(tk.Tk):
    def __init__(self):
        super().__init__()

        self.title("Pipeline Geodésico - GUI")
        self.geometry("1180x860")
        self.minsize(1080, 760)

        self.log_queue = queue.Queue()
        self.worker_thread = None

        self._build_vars()
        self._build_ui()
        self.after(100, self._poll_log_queue)

    # --------------------------------------------------------
    # Variables
    # --------------------------------------------------------
    def _build_vars(self):
        self.vars = {
            "freq": tk.StringVar(value="606,615"),
            "resol": tk.StringVar(value="1"),

            "lat_max": tk.StringVar(value="9.0427396123999593"),
            "lat_min": tk.StringVar(value="3.3594065063999592"),
            "lon_max": tk.StringVar(value="-72.7355829984999929"),
            "lon_min": tk.StringVar(value="-78.4189161044999992"),
            "apertura": tk.StringVar(value="0.326666666666666666666666666666"),

            "si_no": tk.StringVar(value="Si"),

            "puntos": tk.StringVar(value="Obs_Terr_DTU21Gra_Nosotros.txt"),
            "puntos_aero": tk.StringVar(value="Aero_grav_ob_en_IHRF_MA_2000.txt"),
            "listado": tk.StringVar(value="listado.txt"),
            "sigma_file": tk.StringVar(value="Informe_varianza_x_proyectos_f2190.txt"),

            "obs_lat_col": tk.StringVar(value="Lat_Magna"),
            "obs_lon_col": tk.StringVar(value="Long_Magna"),
            "obs_alt_col": tk.StringVar(value="h"),

            "aero_lat_col": tk.StringVar(value="Latitud"),
            "aero_lon_col": tk.StringVar(value="Longitud"),
            "aero_alt_col": tk.StringVar(value="Altura"),

            "tam_bloque": tk.StringVar(value="5000"),

            "obs_filter": tk.StringVar(value="Shannon"),
            "aero_filter": tk.StringVar(value="Cup"),
            "pixel_filter": tk.StringVar(value="Cup"),

            "force_mode": tk.StringVar(value="none"),

            "restore": tk.BooleanVar(value=True),
            "qgeoid": tk.BooleanVar(value=True),

            "case_1VC": tk.BooleanVar(value=True),
            "case_1VCLC": tk.BooleanVar(value=True),
            "case_2VC": tk.BooleanVar(value=False),
            "case_2VCLC": tk.BooleanVar(value=False),
        }

    # --------------------------------------------------------
    # UI
    # --------------------------------------------------------
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

    # --------------------------------------------------------
    # Tabs
    # --------------------------------------------------------
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
            ("Tamaño bloque", "tam_bloque"),
            ("Obs Filter", "obs_filter"),
            ("Aero Filter", "aero_filter"),
            ("Pixel Filter", "pixel_filter"),
            ("Force Mode", "force_mode"),
        ]

        for idx, (label, key) in enumerate(fields):
            r = idx // 2
            c = (idx % 2) * 2
            ttk.Label(parent, text=label).grid(row=r, column=c, sticky="w", padx=5, pady=5)
            ttk.Entry(parent, textvariable=self.vars[key]).grid(row=r, column=c+1, sticky="ew", padx=5, pady=5)

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
        for i in range(4):
            parent.columnconfigure(i, weight=1)

        ttk.Label(parent, text="Observaciones", font=("Segoe UI", 10, "bold")).grid(row=0, column=0, sticky="w", pady=10)
        ttk.Label(parent, text="Aero", font=("Segoe UI", 10, "bold")).grid(row=0, column=2, sticky="w", pady=10)

        obs = [("Lat", "obs_lat_col"), ("Lon", "obs_lon_col"), ("Alt", "obs_alt_col")]
        aero = [("Lat", "aero_lat_col"), ("Lon", "aero_lon_col"), ("Alt", "aero_alt_col")]

        for i, (lab, key) in enumerate(obs, start=1):
            ttk.Label(parent, text=lab).grid(row=i, column=0, sticky="w", padx=5, pady=5)
            ttk.Entry(parent, textvariable=self.vars[key]).grid(row=i, column=1, sticky="ew", padx=5, pady=5)

        for i, (lab, key) in enumerate(aero, start=1):
            ttk.Label(parent, text=lab).grid(row=i, column=2, sticky="w", padx=5, pady=5)
            ttk.Entry(parent, textvariable=self.vars[key]).grid(row=i, column=3, sticky="ew", padx=5, pady=5)

    def _build_process_tab(self, parent):
        parent.columnconfigure(0, weight=1)

        box1 = ttk.LabelFrame(parent, text="Casos", padding=10)
        box1.grid(row=0, column=0, sticky="ew", padx=5, pady=5)

        ttk.Checkbutton(box1, text="1VC", variable=self.vars["case_1VC"]).grid(row=0, column=0, padx=8, pady=5, sticky="w")
        ttk.Checkbutton(box1, text="1VCLC", variable=self.vars["case_1VCLC"]).grid(row=0, column=1, padx=8, pady=5, sticky="w")
        ttk.Checkbutton(box1, text="2VC", variable=self.vars["case_2VC"]).grid(row=1, column=0, padx=8, pady=5, sticky="w")
        ttk.Checkbutton(box1, text="2VCLC", variable=self.vars["case_2VCLC"]).grid(row=1, column=1, padx=8, pady=5, sticky="w")

        box2 = ttk.LabelFrame(parent, text="Procesos", padding=10)
        box2.grid(row=1, column=0, sticky="ew", padx=5, pady=10)

        ttk.Checkbutton(box2, text="Restore", variable=self.vars["restore"]).grid(row=0, column=0, padx=8, pady=5, sticky="w")
        ttk.Checkbutton(box2, text="QGeoid", variable=self.vars["qgeoid"]).grid(row=0, column=1, padx=8, pady=5, sticky="w")

    # --------------------------------------------------------
    # Consola
    # --------------------------------------------------------
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

    # --------------------------------------------------------
    # Helpers
    # --------------------------------------------------------
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
        if self.vars["case_2VC"].get():
            cases.append("2VC")
        if self.vars["case_2VCLC"].get():
            cases.append("2VCLC")

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

            "tam_bloque": self.vars["tam_bloque"].get(),
            "obs_filter": self.vars["obs_filter"].get(),
            "aero_filter": self.vars["aero_filter"].get(),
            "pixel_filter": self.vars["pixel_filter"].get(),

            "cases": cases,
            "force_mode": self.vars["force_mode"].get(),

            "restore": self.vars["restore"].get(),
            "qgeoid": self.vars["qgeoid"].get(),
        }
        return values

    # --------------------------------------------------------
    # Ejecución
    # --------------------------------------------------------
    def run_pipeline(self):
        if self.worker_thread and self.worker_thread.is_alive():
            messagebox.showwarning("En ejecución", "Ya hay un proceso ejecutándose.")
            return

        values = self._collect_values()
        args = build_args_from_form(values)

        self.log("\n" + "=" * 80 + "\n")
        self.log("Ejecutando pipeline...\n")
        self.log("Argumentos:\n")
        self.log(" ".join(args) + "\n\n")

        self.progress.start(10)

        self.worker_thread = threading.Thread(
            target=self._worker_run,
            args=(args,),
            daemon=True
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


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    app = PipelineGUI()
    app.mainloop()