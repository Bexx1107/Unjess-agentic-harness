@echo off
REM ═══════════════════════════════════════════════════════════════
REM  njss — Build Standalone Desktop App
REM  Packages njss into a single .exe using PyInstaller
REM ═══════════════════════════════════════════════════════════════
echo.
echo  Building njss desktop app...
echo  This may take 2-5 minutes on the first run.
echo.

pyinstaller ^
    --name "njss" ^
    --windowed ^
    --noconfirm ^
    --clean ^
    --collect-all nicegui ^
    --collect-all unjess ^
    --hidden-import "engineio.async_drivers.aiohttp" ^
    --hidden-import "tiktoken_ext.openai_public" ^
    --hidden-import "tiktoken_ext" ^
    --hidden-import "google.genai" ^
    --hidden-import "anthropic" ^
    --hidden-import "openai" ^
    --hidden-import "httpx" ^
    --hidden-import "nest_asyncio" ^
    --hidden-import "webview" ^
    --collect-all "rich" ^
    --collect-all "playwright" ^
    --exclude-module torch ^
    --exclude-module torchvision ^
    --exclude-module torchaudio ^
    --exclude-module onnxruntime ^
    --exclude-module cv2 ^
    --exclude-module opencv ^
    --exclude-module selenium ^
    --exclude-module polars ^
    --exclude-module pyarrow ^
    --exclude-module scipy ^
    --exclude-module matplotlib ^
    --exclude-module pandas ^
    --exclude-module numpy ^
    --exclude-module transformers ^
    --exclude-module imageio_ffmpeg ^
    --exclude-module av ^
    --exclude-module pygame ^
    --exclude-module hf_xet ^
    --exclude-module huggingface_hub ^
    --exclude-module datasets ^
    --exclude-module tensorflow ^
    --exclude-module keras ^
    --exclude-module sklearn ^
    --exclude-module IPython ^
    --exclude-module jupyter ^
    --exclude-module notebook ^
    --icon assets\icon.ico ^
    launcher.py

if %ERRORLEVEL% EQU 0 (
    echo.
    echo  Copying assets...
    xcopy /E /I /Y assets dist\njss\assets >nul 2>&1
    echo.
    echo  ══════════════════════════════════════════════
    echo   BUILD SUCCESSFUL!
    echo   Output: dist\njss\njss.exe
    echo  ══════════════════════════════════════════════
    echo.
) else (
    echo.
    echo  BUILD FAILED — check the output above for errors.
    echo.
)
pause
