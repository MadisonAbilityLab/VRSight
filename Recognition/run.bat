@REM Script necessary for some Windows systems to avoid crashes. Run instead of "python main.py" directly.

@echo off
setlocal

@REM :: General PyTorch Settings
set KMP_DUPLICATE_LIB_OK=TRUE
set OMP_DYNAMIC=FALSE
set KMP_BLOCKTIME=0

@REM :: Run the Python script
python main.py

endlocal