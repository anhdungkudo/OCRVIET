#!/usr/bin/env python
"""Kịch bản đóng gói ứng dụng thành file thực thi độc lập (.EXE) cho Windows.

Sử dụng:
    python build_app.py            # Đóng gói dạng thư mục (khởi động cực nhanh, khuyên dùng)
    python build_app.py --onefile  # Đóng gói thành 1 file .exe duy nhất
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Đóng gói ứng dụng OCRVIET thành file .exe")
    parser.add_argument(
        "--onefile",
        action="store_true",
        help="Đóng gói thành 1 file .exe duy nhất (dung lượng nén, khởi động giải nén lần đầu chậm hơn một chút).",
    )
    parser.add_argument(
        "--console",
        action="store_true",
        help="Giữ lại cửa sổ console đen để xem log debug (mặc định là ẩn cửa sổ console).",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Dọn dẹp thư mục build/ và dist/ trước khi đóng gói.",
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent
    dist_dir = project_root / "dist"
    build_dir = project_root / "build"

    # Kiểm tra PyInstaller
    try:
        import PyInstaller  # noqa: F401
    except ImportError:
        print("[LỖI] Chưa cài đặt PyInstaller.")
        print("Vui lòng chạy lệnh: pip install pyinstaller")
        return 1

    if args.clean:
        print("🧹 Đang dọn dẹp các thư mục build cũ...")
        if build_dir.exists():
            shutil.rmtree(build_dir, ignore_errors=True)
        if dist_dir.exists():
            shutil.rmtree(dist_dir, ignore_errors=True)

    app_name = "ChuyenDoiVanBanPDF"

    # Xây dựng danh sách tham số PyInstaller
    cmd = [
        sys.executable,
        "-m",
        "PyInstaller",
        "--name",
        app_name,
        "--noconfirm",
        "--clean",
    ]

    # Chế độ đóng gói
    if args.onefile:
        cmd.append("--onefile")
        print("📦 Chế độ: Đóng gói thành 1 file .exe duy nhất...")
    else:
        cmd.append("--onedir")
        print("📦 Chế độ: Đóng gói dạng thư mục (khởi động tức thì)...")

    # Console mode
    if args.console:
        cmd.append("--console")
    else:
        cmd.append("--windowed")

    # Thu thập data và hidden imports cho các thư viện phức tạp
    cmd.extend([
        "--collect-data", "docx",
        "--collect-data", "pydantic",
        "--collect-data", "rapidocr_onnxruntime",
        "--collect-data", "vietocr",
        "--collect-binaries", "torchvision",
        "--copy-metadata", "gdown",
        "--copy-metadata", "vietocr",
        "--copy-metadata", "rapidocr_onnxruntime",
        "--collect-submodules", "vn_pdf_to_docx",
        "--collect-submodules", "rapidocr_onnxruntime",
        "--hidden-import", "vietocr",
        "--hidden-import", "vietocr.tool.config",
        "--hidden-import", "vietocr.tool.predictor",
        "--hidden-import", "vietocr.tool.translate",
        "--hidden-import", "vietocr.tool.utils",
        "--hidden-import", "vietocr.model.transformerocr",
        "--hidden-import", "vietocr.model.vocab",
        "--hidden-import", "vietocr.model.beam",
        "--hidden-import", "vietocr.model.backbone.cnn",
        "--hidden-import", "vietocr.model.backbone.vgg",
        "--hidden-import", "vietocr.model.backbone.resnet",
        "--hidden-import", "vietocr.model.seqmodel.transformer",
        "--hidden-import", "vietocr.model.seqmodel.seq2seq",
        "--hidden-import", "vietocr.model.seqmodel.convseq2seq",
        "--hidden-import", "fitz",
        "--hidden-import", "pymupdf",
        "--hidden-import", "cv2",
        "--hidden-import", "PIL",
        "--hidden-import", "torch",
        "--hidden-import", "torchvision",
        "--hidden-import", "torchvision._C_stable",
        "--hidden-import", "torchvision.image_stable",
        "--hidden-import", "onnxruntime",
        "--hidden-import", "yaml",
        "--hidden-import", "einops",
        "--hidden-import", "gdown",
        "--hidden-import", "requests",
        "--hidden-import", "tqdm",
        "--hidden-import", "pyclipper",
        "--hidden-import", "shapely",
        "--hidden-import", "pydantic_core",
        "--hidden-import", "PyQt5",
        "--hidden-import", "PyQt5.QtCore",
        "--hidden-import", "PyQt5.QtGui",
        "--hidden-import", "PyQt5.QtWidgets",
    ])
    if sys.platform == "win32":
        sys32_dir = Path(os.environ.get("SystemRoot", r"C:\Windows")) / "System32"
        for dll_name in (
            "msvcp140.dll",
            "msvcp140_1.dll",
            "msvcp140_2.dll",
            "msvcp140_atomic_wait.dll",
            "vcruntime140.dll",
            "vcruntime140_1.dll",
            "vcruntime140_threads.dll",
        ):
            dll_src = sys32_dir / dll_name
            if dll_src.is_file():
                cmd.extend(["--add-binary", f"{dll_src};."])

    # File khởi chạy chính
    entry_point = project_root / "main.py"
    cmd.append(str(entry_point))

    print("🚀 Bắt đầu quá trình biên dịch PyInstaller. Quá trình này có thể mất 1-2 phút...")
    print(f"👉 Câu lệnh: {' '.join(cmd[:6])} ...")

    # Run PyInstaller with a build-time MSVC preload inside PyInstaller's isolated
    # binary-dependency subprocess so PyQt5\Qt5\bin\msvcp140.dll does not shadow
    # System32 and crash onnxruntime (0xC0000005) during Analysis.
    if sys.platform == "win32":
        import PyInstaller.__main__ as pyi_main
        import PyInstaller.building.build_main as pyi_build_main

        orig_isolated_python = pyi_build_main.isolated.Python

        class _PreloadedIsolatedPython(orig_isolated_python):  # type: ignore[misc, valid-type]
            def call(self, function, *args, **kwargs):  # type: ignore[override]
                res = super().call(function, *args, **kwargs)
                if getattr(function, "__name__", "") == "setup":
                    def _preload_msvc() -> None:
                        import ctypes
                        import os
                        from pathlib import Path

                        sys32 = Path(os.environ.get("SystemRoot", r"C:\Windows")) / "System32"
                        for dll in (
                            "vcruntime140.dll",
                            "vcruntime140_1.dll",
                            "vcruntime140_threads.dll",
                            "msvcp140.dll",
                            "msvcp140_1.dll",
                            "msvcp140_2.dll",
                            "msvcp140_atomic_wait.dll",
                        ):
                            p = sys32 / dll
                            if p.is_file():
                                try:
                                    ctypes.WinDLL(str(p))
                                except OSError:
                                    pass

                    super().call(_preload_msvc)
                return res

        pyi_build_main.isolated.Python = _PreloadedIsolatedPython
        os.chdir(str(project_root))
        try:
            pyi_main.run(cmd[3:])
            returncode = 0
        except SystemExit as exc:
            returncode = int(exc.code) if isinstance(exc.code, int) else (0 if exc.code is None else 1)
    else:
        result = subprocess.run(cmd, cwd=str(project_root))
        returncode = result.returncode

    if returncode != 0:
        print(f"\n❌ Đóng gói thất bại với mã lỗi {returncode}.")
        return returncode

    target_output = dist_dir / app_name if not args.onefile else dist_dir

    if target_output.exists():
        # Remove older 2020 MSVC runtime DLLs inside PyQt5\Qt5\bin when newer ones exist
        # in _internal\, preventing 0xC0000005 crashes in onnxruntime/torch.
        internal_dir = target_output / "_internal"
        qt_bin_dir = internal_dir / "PyQt5" / "Qt5" / "bin"
        if not args.onefile and qt_bin_dir.is_dir():
            for stale_dll in (
                "MSVCP140.dll",
                "MSVCP140_1.dll",
                "MSVCP140_2.dll",
                "MSVCP140_ATOMIC_WAIT.dll",
                "VCRUNTIME140.dll",
                "VCRUNTIME140_1.dll",
                "VCRUNTIME140_THREADS.dll",
            ):
                if (internal_dir / stale_dll.lower()).is_file() or (internal_dir / stale_dll).is_file():
                    (qt_bin_dir / stale_dll).unlink(missing_ok=True)

    print("\n" + "=" * 65)
    print("🎉 CHÚC MỪNG! BẠN ĐÃ ĐÓNG GÓI ỨNG DỤNG THÀNH CÔNG!")
    print("=" * 65)
    if args.onefile:
        exe_path = dist_dir / f"{app_name}.exe"
        print(f"📁 File thực thi: {exe_path}")
    else:
        app_folder = dist_dir / app_name
        print(f"📁 Thư mục ứng dụng: {app_folder}")
        print(f"👉 File khởi chạy: {app_folder / f'{app_name}.exe'}")
    print("=" * 65)

    return 0


if __name__ == "__main__":
    sys.exit(main())

