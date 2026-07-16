#include "biblis/toke/fabrica_dados.h"

int main() {
    TokenizadorBPE tok;
    FabricaDados::treinarEsalvar("dados.txt", "merges.txt", "vocab.txt", 8000);
    FabricaDados::carregarTokenizador(&tok, "merges.txt", "vocab.txt");
    FabricaDados::gerar("dados.txt", &tok, "treino.bin", "validacao.bin");
    return 0;
}