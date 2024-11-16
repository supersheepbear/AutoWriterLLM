@echo off
echo Creating Python virtual environment...

:: Download and install pandoc
echo Installing pandoc...
powershell -Command "& {Invoke-WebRequest -Uri 'https://github.com/jgm/pandoc/releases/download/3.1.11/pandoc-3.1.11-windows-x86_64.msi' -OutFile 'pandoc-setup.msi'}"
start /wait msiexec /i pandoc-setup.msi /quiet

:: Download and install wkhtmltopdf
echo Installing wkhtmltopdf...
powershell -Command "& {Invoke-WebRequest -Uri 'https://github.com/wkhtmltopdf/packaging/releases/download/0.12.6-1/wkhtmltox-0.12.6-1.msvc2015-win64.exe' -OutFile 'wkhtmltopdf-setup.exe'}"
start /wait wkhtmltopdf-setup.exe /S /D="C:\Program Files\wkhtmltopdf"

:: Add wkhtmltopdf to PATH (system-wide)
powershell -Command "& {$oldPath = [Environment]::GetEnvironmentVariable('Path', 'Machine'); $newPath = $oldPath + ';C:\Program Files\wkhtmltopdf\bin'; [Environment]::SetEnvironmentVariable('Path', $newPath, 'Machine')}"

:: Create virtual environment
python -m venv venv

:: Activate virtual environment
call venv\Scripts\activate.bat

:: Upgrade pip
python -m pip install --upgrade pip

:: Install requirements
pip install -r requirements.txt

echo.
echo Virtual environment setup complete!
echo To activate the environment, run: venv\Scripts\activate.bat
echo.

pause 