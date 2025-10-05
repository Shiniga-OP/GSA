ARQ=$1
cp /storage/emulated/0/pacotes/gsa-zero/$ARQ.cpp ./
cp -r /storage/emulated/0/pacotes/gsa-zero/biblis ./
cp g.sh /storage/emulated/0/pacotes/gsa-zero/
clang++ $ARQ.cpp -o $ARQ
time ./$ARQ
