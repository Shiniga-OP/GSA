ARQ=$1
OTIMI=$2
cp -rf $CASA/pacotes/gsa/ .
cp g.sh $CASA/pacotes/gsa/
cd gsa
chmod +x compilar.sh
sh compilar.sh $ARQ $OTIMI
