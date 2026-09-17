# PyInstaller spec for keccakviz.  Build on the target OS:
#     pyinstaller keccakviz.spec            (one folder, fastest start)
#     pyinstaller keccakviz.spec -- --onefile  is not supported; use build.py --onefile
import os
import sys

block_cipher = None
onefile = os.environ.get("KECCAKVIZ_ONEFILE") == "1"

# Qt modules we never use; leaving them out roughly halves the bundle.
excludes = [
    "PyQt5.QtWebEngine", "PyQt5.QtWebEngineWidgets", "PyQt5.QtWebEngineCore", "PyQt5.QtWebKit",
    "PyQt5.QtWebKitWidgets", "PyQt5.QtQml", "PyQt5.QtQuick", "PyQt5.QtQuickWidgets", "PyQt5.QtQuick3D",
    "PyQt5.QtMultimedia", "PyQt5.QtMultimediaWidgets", "PyQt5.QtBluetooth", "PyQt5.QtNfc",
    "PyQt5.QtPositioning", "PyQt5.QtLocation", "PyQt5.QtSerialPort", "PyQt5.QtSensors",
    "PyQt5.QtDesigner", "PyQt5.QtHelp", "PyQt5.QtTest", "PyQt5.QtSql", "PyQt5.QtXmlPatterns",
    "PyQt5.QtNetworkAuth", "PyQt5.QtRemoteObjects", "PyQt5.Qt3DCore", "PyQt5.Qt3DRender",
    "tkinter", "matplotlib", "scipy", "pandas", "IPython", "pytest", "setuptools", "pip",
]

a = Analysis(
    ["run_keccakviz.py"],
    pathex=[],
    binaries=[],
    datas=[],
    hiddenimports=["moderngl", "glcontext", "keccakviz", "keccakviz.core", "keccakviz.ui"],
    hookspath=[],
    runtime_hooks=[],
    excludes=excludes,
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)
pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

if onefile:
    exe = EXE(
        pyz, a.scripts, a.binaries, a.zipfiles, a.datas, [],
        name="keccakviz",
        debug=False, bootloader_ignore_signals=False, strip=False, upx=True,
        console=False, disable_windowed_traceback=False,
        icon=None,
    )
else:
    exe = EXE(
        pyz, a.scripts, [],
        exclude_binaries=True,
        name="keccakviz",
        debug=False, bootloader_ignore_signals=False, strip=False, upx=True,
        console=False, disable_windowed_traceback=False,
        icon=None,
    )
    coll = COLLECT(exe, a.binaries, a.zipfiles, a.datas, strip=False, upx=True, name="keccakviz")
