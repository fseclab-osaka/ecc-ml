@rem ----------------------------------------
@rem make vitual environment for this project
@rem ----------------------------------------
pip install virtualenv
rem virtualenv env
virtualenv -p python3.9 venv
call venv\Scripts\activate.bat

python -m pip install --upgrade pip

pip install -r requirements.txt
rem deactivate
