ARQ=$1
OTIMI=$2
COMPILADOR=clang++
if [ ! -d "bin" ]; then
    mkdir bin
fi
$COMPILADOR $ARQ.cpp -o bin/$ARQ $OTIMI -march=native -ffast-math
time ./bin/$ARQ