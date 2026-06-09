$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONHOME = "C:\Users\carlo\AppData\Local\Python\pythoncore-3.14-64"
$PYTHON = "C:\Users\carlo\AppData\Local\Python\pythoncore-3.14-64\python.exe"

& $PYTHON apps/cli/main.py translate @args
