@echo off
REM Streamlit uygulamasını başlatmak için batch script

REM Virtual environment'i aktifleştir
call env\Scripts\activate.bat

REM Streamlit uygulamasını çalıştır
streamlit run app.py

pause
