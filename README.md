# GSA
em desenvolvimento com a biblioteca própria.

## conteúdo:
1. ativações: degrau, sigmoid, tanh, ReLU, GELU, e mais, + deverivadas.
2. tokenizadores: tokenzador sub palavra.
3. utilitários: pesos, atualização de pesos, perda, erro, saída, métricas, matrizes 2D/3D + vetores.
4. camaadas: camada densa

## para testar:

```Cpp
#include "biblis/toke.h"
#include "biblis/camada.h"

int main() {
  testeU(); // testa todos os utilitários
  testeCD(); // testa a camada densa
  testeT(); // testa o tokenizador
  return 0;
}
```

implementação feita do zero, sem bibliotecas de IA prontas.
