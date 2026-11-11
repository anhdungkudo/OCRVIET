@echo off
chcp 65001 >nul
title Phan Mem Chuyen Doi PDF Sang Word (Nghi Dinh 30)

cd /d "%~dp0"

echo =====================================================================
echo    PHẦN MỀM CHUYỂN ĐỔI VĂN BẢN PDF SCAN SANG WORD (.DOCX)
echo    Theo Thể Thức Nghị Định 30/2020/NĐ-CP & Khử Dấu Đỏ
echo =====================================================================
echo Đang khởi động giao diện... Vui lòng đợi trong giây lát...
echo.

set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8

where python >nul 2>&1
if %errorlevel% equ 0 (
    python run_gui.py
    goto check_exit
)

where py >nul 2>&1
if %errorlevel% equ 0 (
    py run_gui.py
    goto check_exit
)

echo.
echo [LỖI] Không tìm thấy Python trên máy tính.
echo Vui lòng cài đặt Python (từ https://python.org) và chọn "Add python.exe to PATH".
pause
exit /b 1

:check_exit
if errorlevel 1 (
    echo.
    echo [LỖI] Ứng dụng đã thoát với mã lỗi.
    echo Vui lòng kiểm tra lại môi trường Python hoặc thông báo lỗi bên trên.
    pause
)
