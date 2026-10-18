# PyInstaller spec for AirDesk (onedir build: fast start, reliable native libs).
# Build:  .venv\Scripts\python -m PyInstaller AirDesk.spec --noconfirm
import os
from PyInstaller.utils.hooks import collect_data_files, collect_submodules
from PyInstaller.utils.win32.versioninfo import (FixedFileInfo, StringFileInfo, StringStruct, StringTable,
                                                  VarFileInfo, VarStruct, VSVersionInfo)

import mediapipe

MP = os.path.dirname(mediapipe.__file__)
import re  # noqa: E402
VERSION_STR = re.search(r'__version__ = "([^"]+)"', open("airdesk/__init__.py", encoding="utf-8").read()).group(1)
VERSION = tuple(int(x) for x in VERSION_STR.split(".")) + (0,)

datas = [
    ("assets/models/hand_landmarker.task", "assets/models"),
    ("assets/models/LICENSE", "assets/models"),
    ("THIRD_PARTY_NOTICES", "."),
    ("README.md", "."),
    ("assets/airdesk.ico", "assets"),
    ("assets/airdesk.png", "assets"),
    ("airdesk/data", "airdesk/data"),
]
datas += collect_data_files("mediapipe", excludes=["**/*.dll", "**/testdata/**"])
binaries = [(os.path.join(MP, "tasks", "c", "libmediapipe.dll"), "mediapipe/tasks/c")]
hidden = collect_submodules("mediapipe.tasks.python") + ["mediapipe.tasks.c", "matplotlib.backends.backend_agg"]

excludes = [
    "tkinter", "PySide6.QtWebEngineCore", "PySide6.QtWebEngineWidgets", "PySide6.QtWebEngineQuick",
    "PySide6.QtWebChannel", "PySide6.QtQml", "PySide6.QtQuick", "PySide6.QtQuickWidgets", "PySide6.Qt3DCore",
    "PySide6.Qt3DRender", "PySide6.QtPdf", "PySide6.QtPdfWidgets", "PySide6.QtMultimedia", "PySide6.QtCharts",
    "PySide6.QtDataVisualization", "PySide6.QtBluetooth", "PySide6.QtSql", "PySide6.QtTest", "PySide6.QtDesigner",
    "pytest", "sounddevice", "IPython", "matplotlib.backends.backend_qtagg", "matplotlib.backends.backend_tkagg",
]

a = Analysis(["launcher.py"], pathex=["."], binaries=binaries, datas=datas, hiddenimports=hidden,
             excludes=excludes, noarchive=False)
pyz = PYZ(a.pure)

version = VSVersionInfo(
    ffi=FixedFileInfo(filevers=VERSION, prodvers=VERSION),
    kids=[StringFileInfo([StringTable("040904B0", [
        StringStruct("CompanyName", "Nguyen Tu Thien Long"),
        StringStruct("FileDescription", "AirDesk - Your Hands. Your Computer."),
        StringStruct("FileVersion", VERSION_STR), StringStruct("ProductName", "AirDesk"),
        StringStruct("ProductVersion", VERSION_STR), StringStruct("OriginalFilename", "AirDesk.exe")])]),
        VarFileInfo([VarStruct("Translation", [1033, 1200])])])

exe = EXE(pyz, a.scripts, [], exclude_binaries=True, name="AirDesk", icon="assets/airdesk.ico",
          console=False, version=version, upx=False, uac_admin=False)
coll = COLLECT(exe, a.binaries, a.datas, name="AirDesk", upx=False)
