@echo off
REM Ativa o ambiente virtual Python do projeto
call .venv\Scripts\activate
REM Instala os pacotes do requirements.txt
python -m pip install -r requirements.txt
REM Abre o prompt interativo já no ambiente virtual
cmd /K
