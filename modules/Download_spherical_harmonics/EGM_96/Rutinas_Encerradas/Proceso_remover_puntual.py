from pathlib import Path
import subprocess
import textwrap


def _matlab_quote(s: str) -> str:
    if s is None or s == "":
        return "[]"

    # MATLAB tolera backslashes dentro de comillas simples,
    # pero se normaliza a "/" para evitar problemas en Windows.
    s = str(s).replace("\\", "/").replace("'", "''")
    return f"'{s}'"


def write_graflab_wrapper_m(
    out_dir,
    *,
    m_filename,
    GM=3986004.415e8,
    R=6378136.3,
    nmin=0,
    nmax="nmaxGGM",
    ellipsoid=1,
    GGM_path=r"C:/dummy.gfc",
    coord=0,
    point_type=1,
    lat_min=0,
    lat_step=0,
    lat_max=0,
    lon_min=0,
    lon_step=0,
    lon_max=0,
    h=0,
    Input_data_path="",
    Output_path=r"C:/salidas",
    Functional_or_Commission=0,
    Functional=(20,),
    fnALFs=1,
    DTM_path="",
    Export_data_txt=1,
    Export_report=1,
    Export_data_mat=0,
    Display_data=0,
    Graphic_format=0,
    Colormap=0,
    Number_of_colors=0,
    DPI=0,
    Status_bar=1,
    Graflab_path="",
):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    m_path = out_dir / m_filename
    func_name = Path(m_filename).stem

    functional_vec = f"[{', '.join(str(int(v)) for v in Functional)}]"
    nmax_str = str(int(nmax)) if isinstance(nmax, (int, float)) else str(nmax)

    m_body = f"""
    function {func_name}()
    % Generado automáticamente desde Python

    GM = {GM};
    R = {R};
    nmin = {int(nmin)};
    nmax = {nmax_str};

    ellipsoid = {int(ellipsoid)};
    GGM_path = {_matlab_quote(GGM_path)};

    coord = {int(coord)};           % 0 = elipsoidal
    point_type = {int(point_type)}; % 0 = grid, 1 = load data

    % Grid:
    lat_min = {lat_min};
    lat_step = {lat_step};
    lat_max = {lat_max};

    lon_min = {lon_min};
    lon_step = {lon_step};
    lon_max = {lon_max};
    h = {h};

    % Load data:
    Input_data_path = {_matlab_quote(Input_data_path)};

    % Point-wise:
    lat = [];
    lon = [];
    h2 = [];

    Output_path = {_matlab_quote(Output_path)};

    Functional_or_Commission = {int(Functional_or_Commission)};
    Functional = {functional_vec};
    fnALFs = {int(fnALFs)};
    DTM_path = {_matlab_quote(DTM_path)};

    Export_data_txt = {int(Export_data_txt)};
    Export_report = {int(Export_report)};
    Export_data_mat = {int(Export_data_mat)};

    Display_data = {int(Display_data)};
    Graphic_format = {int(Graphic_format)};
    Colormap = {int(Colormap)};
    Number_of_colors = {int(Number_of_colors)};
    DPI = {int(DPI)};
    Status_bar = {int(Status_bar)};

    Graflab_path = {_matlab_quote(Graflab_path)};

    if ~isempty(Graflab_path)
        addpath(Graflab_path);
    end

    fprintf('>> Graflab_path   = %s\\n', Graflab_path);
    fprintf('>> GGM_path       = %s\\n', GGM_path);
    fprintf('>> Input_data_path= %s\\n', Input_data_path);
    fprintf('>> Output_path    = %s\\n', Output_path);

    GrafLab('OK', GM, R, nmin, nmax, ellipsoid, GGM_path, ...
            coord, point_type, ...
            lat_min, lat_step, lat_max, lon_min, lon_step, lon_max, h, ...
            Input_data_path, lat, lon, h2, ...
            Output_path, Functional_or_Commission, Functional, fnALFs, DTM_path, ...
            Export_data_txt, Export_report, Export_data_mat, ...
            Display_data, Graphic_format, Colormap, Number_of_colors, DPI, ...
            Status_bar);

    close all force
    end
    """

    m_path.write_text(textwrap.dedent(m_body).lstrip(), encoding="utf-8")
    return str(m_path)


def run_matlab_batch(matlab_exe, mfile_dir, mfile_name):
    mfile_stem = Path(mfile_name).stem
    mdir = str(Path(mfile_dir).resolve()).replace("\\", "/")

    cmd = '"{}" -batch "cd(\'{}\'); {}"'.format(
        matlab_exe,
        mdir,
        mfile_stem,
    )

    print("\n[CMD] ", cmd, "\n")

    res = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    print(res.stdout)
    res.check_returncode()


def remover_puntual(
    *,
    out_dir,
    path_gfc_model1,
    output_model1,
    Nombre_mat,
    lat_max,
    lat_min,
    long_max,
    long_min,
    espaciado,
    DTM,
    matlab_bin=r"C:/Program Files/MATLAB/R2024b/bin/matlab.exe",
    nmin=0,
    nmax1=360,
    ellipsoid=1,
    graflab_path="dependencies/Graflab",
):
    """
    Genera y ejecuta un archivo .m para llamar GrafLab desde MATLAB.

    Parameters
    ----------
    out_dir : str
        Carpeta donde se guardará el archivo .m generado.

    path_gfc_model1 : str
        Ruta al modelo .gfc.

    output_model1 : str
        Carpeta donde GrafLab guardará los resultados.

    Nombre_mat : str
        Nombre del archivo .m que se va a generar.

    lat_max, lat_min, long_max, long_min : float
        Límites geográficos del grid.

    espaciado : float
        Espaciado del grid.

    DTM : str
        Ruta del DTM usado por GrafLab.

    matlab_bin : str
        Ruta al ejecutable de MATLAB.

    nmin, nmax1 : int
        Grados mínimo y máximo.

    ellipsoid : int
        Parámetro de elipsoide usado por GrafLab.

    graflab_path : str
        Carpeta donde está GrafLab.m.

        Ejemplo:
            dependencies/Graflab

        O ruta absoluta:
            D:/2026/GitHub_QgeoidgravSRBF/Qgeoid_grav_SRBF/dependencies/Graflab
    """

    out_dir_abs = Path(out_dir).expanduser().resolve()

    # Asumimos estructura:
    # <root>/(Armonicos|Datos_Tratados|Remover)
    root = out_dir_abs.parent

    def abs_under_root(p):
        p = Path(p)
        return (p if p.is_absolute() else (root / p)).resolve()

    def abs_from_project_or_root(p):
        """
        Intenta resolver rutas relativas.

        Primero las interpreta relativas al directorio actual desde donde
        se ejecuta Python. Si no existen, las interpreta relativas a root.
        """
        p = Path(p)

        if p.is_absolute():
            return p.resolve()

        candidate_project = Path.cwd() / p
        if candidate_project.exists():
            return candidate_project.resolve()

        return (root / p).resolve()

    # Rutas absolutas a GGM y salidas
    gfc1 = abs_under_root(path_gfc_model1)
    out1 = abs_under_root(output_model1)

    # Ruta absoluta a la carpeta donde está GrafLab.m
    graflab_dir = abs_from_project_or_root(graflab_path)

    print(">> ROOT         :", root)
    print(">> out_dir_abs  :", out_dir_abs)
    print(">> GGM 1        :", gfc1)
    print(">> OUT 1        :", out1)
    print(">> GrafLab dir  :", graflab_dir)

    if not graflab_dir.exists():
        raise FileNotFoundError(
            f"No existe la carpeta de GrafLab:\n{graflab_dir}"
        )

    graflab_file = graflab_dir / "GrafLab.m"
    if not graflab_file.exists():
        raise FileNotFoundError(
            f"No se encontró GrafLab.m en:\n{graflab_dir}"
        )

    # Genera el .m
    m1_path = write_graflab_wrapper_m(
        out_dir_abs,
        m_filename=Nombre_mat,
        GM=3986004.415e8,
        R=6378136.3,
        nmin=nmin,
        nmax=nmax1,
        ellipsoid=ellipsoid,
        GGM_path=str(gfc1),
        coord=0,
        point_type=0,
        lat_min=lat_min,
        lat_step=espaciado,
        lat_max=lat_max,
        lon_min=long_min,
        lon_step=espaciado,
        lon_max=long_max,
        h=0,
        Output_path=str(out1),
        Functional_or_Commission=0,
        Functional=(10,),
        fnALFs=1,
        Export_data_txt=1,
        Export_report=1,
        Export_data_mat=0,
        Display_data=0,
        Graphic_format=0,
        Colormap=0,
        Number_of_colors=0,
        DPI=0,
        Status_bar=1,
        DTM_path=DTM,
        Graflab_path=str(graflab_dir),
    )

    print(f".m generado modelo 1: {m1_path}")

    # Ejecuta MATLAB haciendo cd a la carpeta donde está el .m generado
    run_matlab_batch(matlab_bin, out_dir_abs, Path(m1_path).name)
