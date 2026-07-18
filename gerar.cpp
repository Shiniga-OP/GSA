// gerar.cpp
// Carrega um checkpoint ja treinado e gera texto a partir de um prompt, sem re-treinar.
// Uso: ./gerar checkpoint.bin.melhor merges.txt vocab.txt "prompt aqui" [numTokens]
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include "biblis/modelo.h"
#include "biblis/toke/bpe.h"

#define DIM 128
#define N_CAB 4
#define DIM_FF 512
#define N_CAMADAS 4
#define SEQ_MAX 64

static bool carregarCheckpoint(const char* caminho, Modelo* modelo) {
    FILE* a = fopen(caminho, "rb");
    if(!a) { printf("Erro ao abrir checkpoint: %s\n", caminho); return false; }

    for(int c = 0; c < modelo->totalCamadas; c++) {
        float* ptrs[16]; int tams[16];
        modelo->todasCamadas[c]->params(ptrs, tams);
        for(int g = 0; g < modelo->todasCamadas[c]->grupos; g++) {
            int lidos = (int)fread(ptrs[g], sizeof(float), tams[g], a);
            if(lidos != tams[g]) {
                printf("Checkpoint incompleto ou incompativel em %s\n", caminho);
                fclose(a);
                return false;
            }
        }
    }
    fclose(a);
    printf("Checkpoint carregado: %s\n", caminho);
    return true;
}

int main(int argc, char** argv) {
    if(argc < 5) {
        printf("Uso: %s checkpoint.bin merges.txt vocab.txt \"prompt\" [numTokens]\n", argv[0]);
        return 1;
    }
    const char* caminhoCheckpoint = argv[1];
    const char* caminhoMerges = argv[2];
    const char* caminhoVocab = argv[3];
    const char* promptTexto = argv[4];
    int numTokens = argc >= 6 ? atoi(argv[5]) : 150;

    // --- tokenizador ---
    TokenizadorBPE tokenizador;
    tokenizador.carregarMerges(caminhoMerges);
    tokenizador.carregarVocab(caminhoVocab);

    int vocab = tokenizador.vocabTam();
    printf("Vocabulario: %d tokens\n", vocab);

    // --- modelo (mesmas dimensoes usadas no treino) ---
    Modelo modelo(vocab, DIM, N_CAB, DIM_FF, N_CAMADAS, SEQ_MAX);
    modelo.inicializar("xavier"); // sobrescrito pelo checkpoint logo abaixo

    if(!carregarCheckpoint(caminhoCheckpoint, &modelo)) {
        return 1;
    }

    // --- codificar prompt ---
    Vetor<int> idsPrompt; idsPrompt.iniciar();
    tokenizador.codificar(promptTexto, (int)strlen(promptTexto), &idsPrompt);

    if(idsPrompt.tam == 0) {
        printf("Erro: prompt vazio ou nao tokenizavel\n");
        idsPrompt.liberar();
        return 1;
    }

    printf("Prompt codificado: %d tokens\n", idsPrompt.tam);

    // --- gerar ---
    int* gerados = (int*)malloc(numTokens * sizeof(int));
    modelo.gerarGuloso(idsPrompt.dados, idsPrompt.tam, gerados, numTokens);

    int tamTextoGerado;
    char* textoGerado = tokenizador.decodificar(gerados, numTokens, &tamTextoGerado);

    printf("\n=== Geracao a partir de \"%s\" ===\n", promptTexto);
    printf("%s%s\n", promptTexto, textoGerado);

    free(textoGerado);
    free(gerados);
    idsPrompt.liberar();
    return 0;
}