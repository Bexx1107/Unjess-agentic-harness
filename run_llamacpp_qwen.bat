@echo off
set "SERVER_PATH=%LOCALAPPDATA%\Microsoft\WinGet\Packages\ggml.llamacpp_Microsoft.Winget.Source_8wekyb3d8bbwe\llama-server.exe"

if not exist "%SERVER_PATH%" (
    where llama-server.exe >nul 2>&1
    if %ERRORLEVEL% EQU 0 (
        set "SERVER_PATH=llama-server.exe"
    )
)

echo ════════════════════════════════════════════════════════════════
echo  STARTING LLAMA.CPP SERVER (VULKAN GPU ACCELERATED)
echo  Model: Qwen 3.5 9B Instruct (Q4_K_M) via Smoffyy
echo  Port:  8080
echo ════════════════════════════════════════════════════════════════
echo.
echo Note: If this is the first run, llama.cpp will automatically
echo download the model from Hugging Face. This may take a few minutes.
echo.

"%SERVER_PATH%" ^
  -hfr Smoffyy/Qwen3.5-9B-Instruct-Revised-GGUF:Q4_K_M ^
  --port 8080 ^
  -c 32768 ^
  -ngl 99 ^
  --image-min-tokens 1024

pause
