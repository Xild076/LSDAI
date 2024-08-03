import sys
from cx_Freeze import setup, Executable

# Increase the recursion limit
sys.setrecursionlimit(5000)

# List of packages to include in the build
packages = [
    "tkinter", "customtkinter", "matplotlib", "numpy", "sounddevice", "wave", "threading", 
    "os", "pandas", "librosa", "torch", "sklearn", "IPython", "tqdm", "pydub", "nltk", 
    "openai", "datetime", "time", "re", "json", "enum", "ssl", "math"
]

# List of additional files to include in the build
include_files = [
    'emphasis.py', 'gui.py', 'tone_a.py', 'tone_save.py', 'tone_text.py', 'tone.py', 
    'utility.py', 'wpm.py', 'data_path.csv', 'train_path.csv', 'emotion_recog_model.pth', 
    'recorded_audio.wav', 'temp_segment.wav', 'label_encoder.pkl'
]

# Build options
options = {
    'build_exe': {
        'packages': packages,
        'include_files': include_files,
        'excludes': [],
        'include_msvcr': True,
    },
}

# Define the executable
executables = [
    Executable("main.py", base=None)
]

# Setup configuration
setup(
    name="S&D Analysis App",
    version="1.0",
    description="Speech and Diction Analysis App",
    options=options,
    executables=executables
)