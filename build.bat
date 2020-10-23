@echo off

set build=debug
REM set build=debug_fast
REM set build=release

set debug_options=/Od /MTd /Zi
set debug_fast_options=/O2 /MTd /Zi
set release_options=/O2 /MT /DNDEBUG
set compiler_options=/fp:fast /arch:AVX2 /FC /Oi /GR- /EHsc /nologo /std:c++17 /W4 /wd4458 /wd4100 /wd4201 /wd4127 /wd4189
set linker_options=/incremental:no /opt:ref

if %build% == release (
    set config=Release
) else (
    set config=Debug
)

call set compiler_options=%compiler_options% %%%build%_options%%

if not exist build mkdir build
pushd build

cl %compiler_options% /Fe:main.exe ../src/main.cpp /I../libs/voxium/include/ /link %linker_options% ../libs/voxium/%config%/x64/voxium.lib
set cl_error=%ERRORLEVEL%

popd

exit /b %cl_error%
