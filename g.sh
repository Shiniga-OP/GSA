ARQ=$1
cp -rf /storage/emulated/0/pacotes/gsa-zero/ ./
cp g.sh /storage/emulated/0/pacotes/gsa-zero/
cd gsa-zero
clang++ $ARQ.cpp -o $ARQ
time ./$ARQ
