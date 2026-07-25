@echo off
setlocal enabledelayedexpansion

set "GMT_BIN=C:\Programs\gmt6\bin"
set "GS_URL=https://github.com/ArtifexSoftware/ghostpdl-downloads/releases/download/gs10071/gs10071w64.exe"
set "GS_INSTALLER=%TEMP%\gs_installer.exe"

echo ===============================================
echo   Configuracion de GMT (no-conda) + Ghostscript
echo   Ruta GMT: !GMT_BIN!
echo ===============================================
echo.

if not exist "!GMT_BIN!\gmt.exe" (
    echo ERROR: no se encontro gmt.exe en !GMT_BIN!
    echo Verifica la ruta e intenta de nuevo.
    pause
    exit /b 1
)

echo [1/5] Actualizando PATH del usuario con GMT...
for /f "tokens=2,*" %%A in ('reg query "HKCU\Environment" /v Path 2^>nul') do set "USER_PATH=%%B"

echo !USER_PATH! | find /I "!GMT_BIN!" >nul
if !errorlevel! equ 0 (
    echo La ruta de GMT ya estaba en el PATH del usuario, no se duplica.
) else (
    if defined USER_PATH (
        setx Path "!USER_PATH!;!GMT_BIN!" >nul
    ) else (
        setx Path "!GMT_BIN!" >nul
    )
    echo PATH actualizado con GMT.
)

echo.
echo [2/5] Configurando GMT_LIBRARY_PATH ^(requerido por pygmt^)...
setx GMT_LIBRARY_PATH "!GMT_BIN!" >nul

echo.
echo [3/5] Verificando si Ghostscript ya esta disponible...
where gswin64c.exe >nul 2>nul
if !errorlevel! equ 0 (
    echo Ghostscript ya esta disponible en el PATH, se omite instalacion.
    goto :verificar
)

echo.
echo [4/5] Descargando e instalando Ghostscript ^(instalador oficial^)...
powershell -Command "Invoke-WebRequest -Uri '!GS_URL!' -OutFile '!GS_INSTALLER!'"

if not exist "!GS_INSTALLER!" (
    echo ERROR: no se pudo descargar el instalador de Ghostscript.
    echo Verifica tu conexion o descarga manualmente desde:
    echo   Ghostscript : Releases
    pause
    exit /b 1
)

echo Ejecutando instalador en modo silencioso...
"!GS_INSTALLER!" /S

echo Esperando a que finalice la instalacion...
timeout /t 15 /nobreak >nul

echo.
echo [5/5] Buscando gswin64c.exe para agregarlo al PATH...
set "GS_EXE="
for /f "delims=" %%F in ('where gswin64c.exe 2^>nul') do if not defined GS_EXE set "GS_EXE=%%F"

if not defined GS_EXE (
    for /f "delims=" %%F in ('dir /s /b "C:\Program Files\gs\gswin64c.exe" 2^>nul') do if not defined GS_EXE set "GS_EXE=%%F"
)
if not defined GS_EXE (
    for /f "delims=" %%F in ('dir /s /b "C:\Program Files (x86)\gs\gswin64c.exe" 2^>nul') do if not defined GS_EXE set "GS_EXE=%%F"
)

if not defined GS_EXE (
    echo.
    echo No se pudo localizar gswin64c.exe automaticamente.
    echo Puede que la instalacion silenciosa no haya terminado a tiempo.
    echo Espera unos segundos e intenta ejecutar este script de nuevo,
    echo o busca la ruta manualmente con:
    echo   Get-ChildItem -Path "C:\" -Filter "gswin64c.exe" -Recurse -ErrorAction SilentlyContinue
    pause
    exit /b 0
)

for %%F in ("!GS_EXE!") do set "GS_BIN=%%~dpF"
if "!GS_BIN:~-1!"=="\" set "GS_BIN=!GS_BIN:~0,-1!"

echo Ghostscript encontrado en: !GS_BIN!

for /f "tokens=2,*" %%A in ('reg query "HKCU\Environment" /v Path 2^>nul') do set "USER_PATH=%%B"

echo !USER_PATH! | find /I "!GS_BIN!" >nul
if !errorlevel! equ 0 (
    echo La ruta de Ghostscript ya estaba en el PATH del usuario, no se duplica.
) else (
    setx Path "!USER_PATH!;!GS_BIN!" >nul
    echo PATH actualizado con Ghostscript.
)

:verificar
echo.
echo ===============================================
echo Listo. CIERRA esta terminal ^(y VS Code si lo usas^)
echo y abre una nueva para que los cambios surtan efecto.
echo.
echo Luego verifica con:
echo   gmt --version
echo   gswin64c --version
echo   python -c "import pygmt; pygmt.show_versions()"
echo ===============================================
