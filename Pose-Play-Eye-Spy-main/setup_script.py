import sys
from cx_Freeze import setup, Executable

# Dependencies are automatically detected, but it might need fine-tuning.
# "packages": ["os"] is used as an example only
build_exe_options = {
    "packages": ["os", "cv2", "matplotlib", "pyautogui", "time", "math", "mediapipe"],
    "excludes": ["module_to_exclude", "another_module"],
}


base = None
if sys.platform == "win32":
    base = "Win32GUI"

setup(
    name="guifoo",
    version="0.1",
    description="My GUI application!",
    options={"build_exe": build_exe_options},
    executables=[Executable("app.py", base=base)],
)
