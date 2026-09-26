cls

cl.exe /c /EHsc main.cpp /I "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.5\include" /Fo:build/main.obj

link.exe build/main.obj /LIBPATH:"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.5\lib\x64" cudart.lib /out:build/main.exe

cd build

main.exe