@echo off
title WAVE - Earth Engine Authentication
echo Authenticating to Google Earth Engine...
echo.
echo A web browser will open. Sign in with your Earth Engine Google account
echo and approve access. When it finishes, your credentials are cached and the
echo WAVE app will reuse them (no need to sign in again unless they expire).
echo.

REM Run from the folder this script lives in.
cd /d "%~dp0"

set "ENV_NAME=wave_env"
set "EE_CMD=import ee; ee.Authenticate(auth_mode='localhost'); print('Authentication complete. Credentials cached.')"

REM 1) Prefer mamba/conda already on PATH.
where mamba >nul 2>nul && (
    mamba run -n %ENV_NAME% python -c "%EE_CMD%"
    goto :end
)
where conda >nul 2>nul && (
    conda run -n %ENV_NAME% python -c "%EE_CMD%"
    goto :end
)

REM 2) Fall back to common per-user install locations (miniforge / miniconda / anaconda).
for %%B in (
    "%LOCALAPPDATA%\miniforge3\condabin\mamba.bat"
    "%USERPROFILE%\miniforge3\condabin\mamba.bat"
    "%LOCALAPPDATA%\miniforge3\condabin\conda.bat"
    "%USERPROFILE%\miniconda3\condabin\conda.bat"
    "%USERPROFILE%\anaconda3\condabin\conda.bat"
    "%LOCALAPPDATA%\miniconda3\condabin\conda.bat"
    "%LOCALAPPDATA%\anaconda3\condabin\conda.bat"
) do (
    if exist "%%~B" (
        call "%%~B" run -n %ENV_NAME% python -c "%EE_CMD%"
        goto :end
    )
)

echo.
echo ERROR: Could not find conda or mamba.
echo   - Install Miniforge/Miniconda, or add conda to your PATH, then
echo   - create the environment with:  conda env create -f wave_environment.yml
echo     (on Windows use wave_environment_win.yml)
echo.

:end
pause
