# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.utils.hooks import collect_data_files
from PyInstaller.utils.hooks import collect_dynamic_libs
from PyInstaller.utils.hooks import collect_submodules
from PyInstaller.utils.hooks import copy_metadata

datas = []
binaries = [('C:/WINDOWS/System32/msvcp140.dll', '.'), ('C:/WINDOWS/System32/msvcp140_1.dll', '.'), ('C:/WINDOWS/System32/msvcp140_2.dll', '.'), ('C:/WINDOWS/System32/msvcp140_atomic_wait.dll', '.'), ('C:/WINDOWS/System32/vcruntime140.dll', '.'), ('C:/WINDOWS/System32/vcruntime140_1.dll', '.'), ('C:/WINDOWS/System32/vcruntime140_threads.dll', '.')]
hiddenimports = ['vietocr', 'vietocr.tool.config', 'vietocr.tool.predictor', 'vietocr.tool.translate', 'vietocr.tool.utils', 'vietocr.model.transformerocr', 'vietocr.model.vocab', 'vietocr.model.beam', 'vietocr.model.backbone.cnn', 'vietocr.model.backbone.vgg', 'vietocr.model.backbone.resnet', 'vietocr.model.seqmodel.transformer', 'vietocr.model.seqmodel.seq2seq', 'vietocr.model.seqmodel.convseq2seq', 'dotenv', 'fitz', 'pymupdf', 'cv2', 'PIL', 'torch', 'torchvision', 'torchvision._C_stable', 'torchvision.image_stable', 'onnxruntime', 'yaml', 'einops', 'gdown', 'requests', 'tqdm', 'pyclipper', 'shapely', 'pydantic_core', 'PyQt5', 'PyQt5.QtCore', 'PyQt5.QtGui', 'PyQt5.QtWidgets']
datas += collect_data_files('docx')
datas += collect_data_files('pydantic')
datas += collect_data_files('rapidocr_onnxruntime')
datas += collect_data_files('vietocr')
datas += copy_metadata('gdown')
datas += copy_metadata('vietocr')
datas += copy_metadata('rapidocr_onnxruntime')
binaries += collect_dynamic_libs('torchvision')
hiddenimports += collect_submodules('google.genai')
hiddenimports += collect_submodules('vn_pdf_to_docx')
hiddenimports += collect_submodules('rapidocr_onnxruntime')


a = Analysis(
    ['E:/code/OCRVIET/main.py'],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='ChuyenDoiVanBanPDF',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='ChuyenDoiVanBanPDF',
)
