@echo off
setlocal enabledelayedexpansion

REM =========================
REM Basic args (irace format)
REM =========================
set CONFIG_ID=%1
set INSTANCE_ID=%2
set SEED=%3

REM All other parameters are the candidate parameters to be passed to program
if "%3"=="" (
    echo ERROR: Not enough parameters
    exit /b 1
)

shift
shift
shift

set CONFIG_PARAMS=%*

REM =========================
REM executable file
REM =========================
set EXE=runner.exe

REM =========================
REM standard output/error
REM =========================
set STDOUT=c%CONFIG_ID%-%INSTANCE_ID%-%SEED%.stdout
set STDERR=c%CONFIG_ID%-%INSTANCE_ID%-%SEED%.stderr

REM =========================
REM call command line
REM =========================
%EXE% --seed %SEED% %CONFIG_PARAMS%
