## Installation on Windows

The binary files in `roypy/` are for Python 3.10 on Windows. If you want to build on your version of Python:

First, install Visual Studio, CMake, and swig. Make sure they are in your PATH or install them with Scoop / winget:

```powershell
winget install cmake
scoop install swig
```

Then, activate your virtual environment, in which you should install `numpy`, `matplotlib`, and `pywin32`. To make sure it compiles windows, define `ssize_t` in `swig/roypy.h` by

```c++
#if defined(_MSC_VER)
    #include <BaseTsd.h>
    typedef SSIZE_T ssize_t;
#endif
```

or if that doesn't work, try:

```c++
#if !defined(ssize_t) && !defined(__MINGW32__)
    #if defined(_WIN64)
        typedef __int64 ssize_t; 
    #else
        typedef long ssize_t;
    #endif
#endif
```

With the correct python path set, run

```powershell
cmake -S .\swig -B .\_build -G "Visual Studio 16 2019" -Droyale_USE_NUMPY_IN_ROYPY=on
```
You may also include `-T ClangCL` if you prefer. You may also see if other versions of Visual Studio work, and as far as I tested, VS2019 should work well with Python 3.10 and Python 3.11. 

Build and copy the built files to your project path
```powershell
cmake --build .\_build\ --config Release
cp .\_build\bin\* .\
```
you might be seeing some warnings, but it's OK to ignore them. Now you may try if everything works by
```powershell
python .\sample_opencv.py
```