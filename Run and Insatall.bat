@echo off
REM Ensure Python and required libraries are installed
echo Installing required Python libraries...
pip install opencv-python mediapipe numpy

REM Run the Bell's Palsy GUI application
echo Starting the Bell's Palsy application...
python Bells_Palsy.py

REM Optional: Pause to keep the command window open after execution
pause
