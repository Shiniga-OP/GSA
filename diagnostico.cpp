// diagnostico.cpp
// Le treino.bin diretamente e decodifica algumas sequencias de volta pra texto,
// sem envolver o modelo. Serve pra confirmar exatamente o que estava nos dados
// de treino, sem depender de nenhuma suposicao sobre o pipeline.
// Uso: ./diagnostico treino.bin merges.txt vocab.txt [numSequenciasMostrar]
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include "biblis/toke/bpe.h"
#include "biblis/toke/fabrica_dados.h"

int main(int argc, char** argv) {
    if(argc < 4) {
        printf("Uso: %s treino.bin merges.txt vocab.txt [numSequenciasMostrar]\n", argv[0]);
        return 1;
    }
    const char* caminhoBin = argv[1];
    const char* caminhoMerges = argv[2];
    const char* caminhoVocab = argv[3];
    int numMostrar = argc >= 5 ? atoi(argv[4]) : 10;

    FILE* a = fopen(caminhoBin, "rb");
    if(!a) { printf("Erro ao abrir %s\n", caminhoBin); return 1; }

    CabecalhoDadosTreino cab;
    if(fread(&cab, sizeof(CabecalhoDadosTreino), 1, a) != 1) {
        printf("Erro ao ler cabecalho\n");
        return 1;
    }
    if(cab.numeroMagico != FABRICA_NUMERO_MAGICO) {
        printf("Numero magico invalido - arquivo corrompido ou formato errado\n");
        return 1;
    }

    printf("Cabecalho: vocab=%d tamJanela=%d numSequencias=%d\n",
           cab.tamVocab, cab.tamJanela, cab.numSequencias);

    TokenizadorBPE tokenizador;
    tokenizador.carregarMerges(caminhoMerges);
    tokenizador.carregarVocab(caminhoVocab);
    printf("Tokenizador carregado: %d tokens no vocab\n", tokenizador.vocabTam());

    if(tokenizador.vocabTam() != cab.tamVocab) {
        printf("\n*** ALERTA: vocabTam do tokenizador (%d) != tamVocab do cabecalho (%d) ***\n",
               tokenizador.vocabTam(), cab.tamVocab);
        printf("*** Os merges/vocab atuais NAO sao os mesmos usados para gerar este .bin ***\n\n");
    }

    int32_t* janela = (int32_t*)malloc(cab.tamJanela * sizeof(int32_t));
    int passo = cab.numSequencias / numMostrar;
    if(passo < 1) passo = 1;

    for(int i = 0; i < numMostrar && i * passo < cab.numSequencias; i++) {
        int idx = i * passo;
        fseek(a, sizeof(CabecalhoDadosTreino) + (size_t)idx * cab.tamJanela * sizeof(int32_t), SEEK_SET);
        int lidos = (int)fread(janela, sizeof(int32_t), cab.tamJanela, a);
        if(lidos != cab.tamJanela) { printf("Erro lendo sequencia %d\n", idx); continue; }

        // converte int32_t -> int pra decodificar (so os primeiros tamJanela-1, sem o alvo extra)
        int* ids = (int*)malloc((cab.tamJanela - 1) * sizeof(int));
        for(int k = 0; k < cab.tamJanela - 1; k++) ids[k] = (int)janela[k];

        int tamTexto;
        char* texto = tokenizador.decodificar(ids, cab.tamJanela - 1, &tamTexto);

        printf("--- sequencia %d (de %d) ---\n%s\n\n", idx, cab.numSequencias, texto);

        free(texto);
        free(ids);
    }

    free(janela);
    fclose(a);
    return 0;
}