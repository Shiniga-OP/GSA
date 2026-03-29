# GSA

## o que é:
GSA é a minha arquitetura de Inteligencia Artificial economica.

## como funciona:
baseada na biologia, a IA que estou produzindo aprenderá lentamente 
gastando o minimo de recursos possivel, ela irá aprender com interações 
como uma criança humana, e se lembrar de experiencias como uma pessoa real. 
Para isso, desenvolvi métodos para armazenamento eficiente de dados, e um processamento neural 
fora do padrão industrial, como uma IA sem filtros, a **ALVA** será autonoma em suas respostas, 
em sua fase inicial, respondendo por demanda, mas logo logo será capaz de tomar a iniciativa e
responder por conta propria.

## estado atual:
em desenvolvimento.

## estrutura:
```Sh
~/gsa $ find . -name "*.h"
./biblis/atencao.h
./biblis/ativas.h
./biblis/camadas.h
./biblis/memoria.h
./biblis/otimizadores.h
./biblis/toke.h
./biblis/util.h
./biblis/otimis/otimizador.h
./biblis/otimis/adam.h
./biblis/otimis/sgd.h
./biblis/otimis/adagrad.h
./biblis/otimis/rmsprop.h
./biblis/otimis/adadelta.h
./biblis/otimis/nesterov.h
./biblis/otimis/adamw.h
./biblis/camadas/atencao.h
./biblis/camadas/densa.h
./biblis/camadas/camada.h
./biblis/camadas/dropout.h
./biblis/camadas/lotenorm.h
./biblis/camadas/conv2d.h
./biblis/camadas/maxreuso2d.h
./biblis/camadas/flatten.h
./biblis/camadas/norm.h
./biblis/camadas/perda.h
./biblis/camadas/embedding.h
```
**implementação feita do zero, sem bibliotecas de IA externas.**