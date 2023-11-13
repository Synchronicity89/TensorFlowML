@echo off
echo Checking system for TensorFlow GPU prerequisites...
echo.

echo [1/4] Checking Python installation...
python --version
if %errorlevel% neq 0 (
    echo Python is not installed or not added to PATH.
    goto end
) else (
    echo Python is installed.
)
echo.

echo [2/4] Checking TensorFlow installation...
python -c "import tensorflow as tf; print('TensorFlow version:', tf.__version__)"
if %errorlevel% neq 0 (
    echo TensorFlow is not installed in this Python environment.
    goto end
) else (
    echo TensorFlow is installed.
)
echo.

echo [3/4] Checking CUDA and cuDNN installation...
where cudart64_*.dll
if %errorlevel% neq 0 (
    echo CUDA is not installed or not added to PATH.
    goto end
) else (
    echo CUDA is installed.
)
echo.

echo [4/4] Checking GPU availability in TensorFlow...
python -c "import tensorflow as tf; gpus = tf.config.list_physical_devices('GPU'); print('Num GPUs Available:', len(gpus));"
if %errorlevel% neq 0 (
    echo Failed to run GPU check in TensorFlow.
    goto end
) else (
    echo GPU check completed.
)
echo.

:end
echo.
echo Check complete.
pause
