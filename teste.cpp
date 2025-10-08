#include "biblis/camadas.h"
#include "biblis/toke.h"

int main() {
    testeU(); // testa todos os utilitários
    testeCD(); // testa a camada densa
    testeCC(); // testa a camada convolucional
    testeCA(); // testa a camada atenção
    testeT(); // testa o tokenizador
    return 0;
}