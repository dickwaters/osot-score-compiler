@echo off
chcp 65001 >nul
echo ====================================================
echo   OSOT 工具 03：v0.2 播音分谱工作台与 TTS 协议编译器
echo ====================================================
echo.
set "PY_BIN=D:\otos\osot-annotator\.venv\Scripts\python.exe"
if not exist "%PY_BIN%" set "PY_BIN=python"

set /p JSON_FILE=请拖入 OSOT JSON 文件（直接回车运行 Demo 并打开 v0.2 工作台）: 
if "%JSON_FILE%"=="" (
    "%PY_BIN%" "%~dp0osot_score_compiler.py" --demo --out-dir "%~dp0demo_compile_out"
    start "" "%~dp0osot_v02_workbench.html"
    pause
    exit /b 0
)

set /p AUTO_FILE=可选：拖入对应机器初标 JSON 以生成 DPO 训练对（无则直接回车）: 
if "%AUTO_FILE%"=="" (
    "%PY_BIN%" "%~dp0osot_score_compiler.py" --json %JSON_FILE% --out-dir "%~dp0compiled_out"
) else (
    "%PY_BIN%" "%~dp0osot_score_compiler.py" --json %JSON_FILE% --auto-json %AUTO_FILE% --out-dir "%~dp0compiled_out"
)
echo.
echo 编译完成！输出位于 %~dp0compiled_out
pause
