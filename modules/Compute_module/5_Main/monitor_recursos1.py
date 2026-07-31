# monitor_recursos.py
import time
import threading
from pathlib import Path
import numpy as np
import psutil

import plotly.graph_objects as go
from plotly.subplots import make_subplots

try:
    import pynvml
    pynvml.nvmlInit()
    NVML_OK = True
except Exception:
    NVML_OK = False


class MonitorRecursos:
    def __init__(self, intervalo=1.0):
        self.intervalo = intervalo
        self.data = []
        self.step_marks = []

        self._running = False
        self._thread = None

    # ----------------------------
    # Control
    # ----------------------------
    def start(self):
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join()

    def mark_step(self, nombre):
        self.step_marks.append((time.time(), nombre))

    # ----------------------------
    # Loop de monitoreo
    # ----------------------------
    def _loop(self):
        t0 = time.time()

        while self._running:
            t = time.time() - t0

            cpu = psutil.cpu_percent()
            ram = psutil.virtual_memory().percent

            gpu = np.nan
            vram = np.nan

            if NVML_OK:
                try:
                    h = pynvml.nvmlDeviceGetHandleByIndex(0)
                    util = pynvml.nvmlDeviceGetUtilizationRates(h)
                    mem = pynvml.nvmlDeviceGetMemoryInfo(h)
                    gpu = util.gpu
                    vram = 100 * mem.used / mem.total
                except Exception:
                    pass

            self.data.append([t, cpu, ram, gpu, vram])
            time.sleep(self.intervalo)

    # ----------------------------
    # Guardar NPY
    # ----------------------------
    def save_npy(self, ruta: Path):
        ruta.parent.mkdir(parents=True, exist_ok=True)
        np.save(ruta, {
            "data": np.array(self.data),
            "steps": self.step_marks
        })

    # ----------------------------
    # Guardar HTML (GARANTIZADO)
    # ----------------------------
    def save_html(self, ruta: Path):
        if len(self.data) == 0:
            raise RuntimeError("MonitorRecursos: no hay datos para graficar")

        ruta.parent.mkdir(parents=True, exist_ok=True)

        arr = np.array(self.data)
        t = arr[:, 0]

        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            subplot_titles=("CPU / RAM", "GPU / VRAM")
        )

        # CPU / RAM
        fig.add_trace(go.Scatter(x=t, y=arr[:, 1], name="CPU %"), row=1, col=1)
        fig.add_trace(go.Scatter(x=t, y=arr[:, 2], name="RAM %"), row=1, col=1)

        # GPU / VRAM
        fig.add_trace(go.Scatter(x=t, y=arr[:, 3], name="GPU %"), row=2, col=1)
        fig.add_trace(go.Scatter(x=t, y=arr[:, 4], name="VRAM %"), row=2, col=1)

        # Pasos
        for ts, nombre in self.step_marks:
            fig.add_vline(
                x=ts - self.step_marks[0][0],
                line_dash="dot",
                line_color="black",
                annotation_text=nombre,
                annotation_position="top"
            )

        fig.update_layout(
            height=700,
            title="Informe uso de recursos",
            xaxis_title="Tiempo [s]",
            legend=dict(orientation="h")
        )

        fig.write_html(
            ruta,
            full_html=True,
            include_plotlyjs="inline"
        )
