@echo off
chcp 65001 >nul
title Dong Goi Ung Dung Sang File EXE

cd /d "%~dp0"

echo =====================================================================
echo    CÔNG CỤ TỰ ĐỘNG ĐÓNG GÓI ỨNG DỤNG SANG FILE .EXE CHO WINDOWS
echo =====================================================================
echo.
echo Bạn muốn đóng gói theo hình thức nào?
echo   [1] Đóng gói dạng thư mục (Khởi động cực nhanh, ổn định nhất - Khuyên dùng)
echo   [2] Đóng gói thành 1 file .EXE duy nhất (Gọn nhẹ nhưng giải nén lần đầu lâu hơn một chút)
echo.
set /p choice="Nhập lựa chọn của bạn (1 hoặc 2, mặc định là 1): "

if "%choice%"=="2" (
    echo.
    echo Đang đóng gói thành 1 file .exe duy nhất...
    python build_app.py --onefile
) else (
    echo.
    echo Đang đóng gói dạng thư mục ứng dụng...
    python build_app.py
)

echo.
pause

