// treino.cpp
// Loop de treino real do modelo, usando os binarios gerados por fabrica_dados.h.
// Uso esperado: ./treino treino.bin validacao.bin merges.txt vocab.bin checkpoint.bin
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <ctime>
#include "biblis/modelo.h"
#include "biblis/otimis/adamw.h"
#include "biblis/toke/bpe.h"
#include "biblis/toke/fabrica_dados.h"

#define TAM_BATCH 8
#define PASSOS_ENTRE_VALIDACAO 200
#define DIM 128
#define N_CAB 4
#define DIM_FF 512
#define N_CAMADAS 4
#define SEQ_MAX 64
#define TAXA_MAX 3e-4f
#define TAXA_MIN 3e-5f
#define PASSOS_AQUECIMENTO 100

// ---------------------------------------------------------------------------
// leitura dos binarios de dados de treino/validacao (formato de fabrica_dados.h)
// ---------------------------------------------------------------------------
struct DadosTreino {
    int32_t tamVocab;
    int32_t tamJanela; // 65 (64 entrada + 1 alvo)
    int32_t numSequencias;
    int32_t* dados; // [numSequencias * tamJanela]

    bool carregar(const char* caminho) {
        FILE* a = fopen(caminho, "rb");
        if(!a) { printf("Erro ao abrir %s\n", caminho); return false; }

        CabecalhoDadosTreino cab;
        if(fread(&cab, sizeof(CabecalhoDadosTreino), 1, a) != 1) {
            printf("Erro ao ler cabecalho de %s\n", caminho);
            fclose(a);
            return false;
        }
        if(cab.numeroMagico != FABRICA_NUMERO_MAGICO) {
            printf("Numero magico invalido em %s (arquivo corrompido ou formato errado)\n", caminho);
            fclose(a);
            return false;
        }

        tamVocab = cab.tamVocab;
        tamJanela = cab.tamJanela;
        numSequencias = cab.numSequencias;

        int totalInts = numSequencias * tamJanela;
        dados = (int32_t*)malloc(totalInts * sizeof(int32_t));
        int lidos = (int)fread(dados, sizeof(int32_t), totalInts, a);
        fclose(a);

        if(lidos != totalInts) {
            printf("Erro: esperava %d ints, leu %d em %s\n", totalInts, lidos, caminho);
            free(dados);
            return false;
        }

        printf("Carregado %s: %d sequencias, janela=%d, vocab=%d\n",
               caminho, numSequencias, tamJanela, tamVocab);
        return true;
    }

    // ponteiro pra sequencia i (tamJanela ints: 64 entrada + 1 alvo final)
    const int32_t* sequencia(int i) const {
        return dados + (size_t)i * tamJanela;
    }

    void liberar() { free(dados); }
};

// converte int32_t* pra int* (buffers de entrada/alvo esperados por Modelo)
static void montarEntradaAlvo(const int32_t* janela, int tamJanela, int* entrada, int* alvo) {
    int seq = tamJanela - 1; // 64
    for(int i = 0; i < seq; i++) {
        entrada[i] = (int)janela[i];
        alvo[i] = (int)janela[i + 1];
    }
}

// ---------------------------------------------------------------------------
// checkpoint: grava params() de todas as camadas, sequencialmente, em binario
// ---------------------------------------------------------------------------
static void salvarCheckpoint(const char* caminho, Modelo* modelo) {
    FILE* a = fopen(caminho, "wb");
    if(!a) { printf("Erro ao salvar checkpoint em %s\n", caminho); return; }

    for(int c = 0; c < modelo->totalCamadas; c++) {
        float* ptrs[16]; int tams[16];
        modelo->todasCamadas[c]->params(ptrs, tams);
        for(int g = 0; g < modelo->todasCamadas[c]->grupos; g++) {
            fwrite(ptrs[g], sizeof(float), tams[g], a);
        }
    }
    fclose(a);
    printf("Checkpoint salvo: %s\n", caminho);
}

static bool carregarCheckpoint(const char* caminho, Modelo* modelo) {
    FILE* a = fopen(caminho, "rb");
    if(!a) return false; // ausencia de checkpoint nao e erro: treino comeca do zero

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

// ---------------------------------------------------------------------------
// avalia perda media sobre um numero limitado de sequencias de validacao
// (nao percorre tudo a cada checagem, pra nao pesar no tempo total de treino)
// ---------------------------------------------------------------------------
static float avaliarValidacao(Modelo* modelo, DadosTreino* val, int maxSequencias) {
    int n = val->numSequencias < maxSequencias ? val->numSequencias : maxSequencias;
    int entrada[SEQ_MAX], alvo[SEQ_MAX];
    float somaPerda = 0.0f;

    modelo->defSeq(SEQ_MAX);
    for(int i = 0; i < n; i++) {
        montarEntradaAlvo(val->sequencia(i), val->tamJanela, entrada, alvo);
        modelo->prop(entrada);
        somaPerda += modelo->perdaCrossEntropy(alvo);
    }
    return somaPerda / (float)n;
}

int main(int argc, char** argv) {
    if(argc < 6) {
        printf("Uso: %s treino.bin validacao.bin merges.txt vocab.bin checkpoint.bin\n", argv[0]);
        return 1;
    }
    const char* caminhoTreino = argv[1];
    const char* caminhoValidacao = argv[2];
    const char* caminhoMerges = argv[3];
    const char* caminhoVocab = argv[4];
    const char* caminhoCheckpoint = argv[5];

    char caminhoMelhor[512];
    snprintf(caminhoMelhor, sizeof(caminhoMelhor), "%s.melhor", caminhoCheckpoint);

    // --- 1. carregar dados ---
    DadosTreino treino, validacao;
    if(!treino.carregar(caminhoTreino)) return 1;
    if(!validacao.carregar(caminhoValidacao)) return 1;

    if(treino.tamVocab != validacao.tamVocab) {
        printf("Vocab de treino (%d) e validacao (%d) nao batem\n", treino.tamVocab, validacao.tamVocab);
        return 1;
    }
    if(treino.tamJanela != SEQ_MAX + 1) {
        printf("tamJanela (%d) nao bate com SEQ_MAX+1 (%d)\n", treino.tamJanela, SEQ_MAX + 1);
        return 1;
    }

    // --- 2. tokenizador (so pra geracao de exemplo ao final) ---
    TokenizadorBPE tokenizador;
    tokenizador.carregarMerges(caminhoMerges);
    tokenizador.carregarVocab(caminhoVocab);

    // --- 3. modelo + otimizador ---
    Modelo modelo(treino.tamVocab, DIM, N_CAB, DIM_FF, N_CAMADAS, SEQ_MAX);
    modelo.inicializar("xavier");

    bool retomado = carregarCheckpoint(caminhoCheckpoint, &modelo);
    if(!retomado) printf("Nenhum checkpoint encontrado, treinando do zero.\n");

    AdamW otim;
    otim.iniciar(modelo.todasCamadas, modelo.totalCamadas, TAXA_MAX);
    printf("Modelo: %d parametros, %d grupos\n", otim.totalN, otim.nGrupos);

    // --- 4. loop de treino ---
    modelo.defSeq(SEQ_MAX);
    int entrada[SEQ_MAX], alvo[SEQ_MAX];

    int totalPassosBatch = treino.numSequencias / TAM_BATCH;
    printf("Iniciando treino: %d sequencias, batch=%d, %d passos por epoca\n",
           treino.numSequencias, TAM_BATCH, totalPassosBatch);

    AgendadorCosseno agendador;
    agendador.taxaMax = TAXA_MAX;
    agendador.taxaMin = TAXA_MIN;
    agendador.passosTotal = totalPassosBatch;
    agendador.aquecimento = PASSOS_AQUECIMENTO;

    float melhorPerdaValidacao = 1e30f; // nenhuma validacao ainda

    srand((unsigned int)time(nullptr));
    time_t inicioTreino = time(nullptr);

    for(int passo = 0; passo < totalPassosBatch; passo++) {
        otim.taxa = agendador.calcular(passo);

        modelo.zerarGrad();
        float somaPerdaBatch = 0.0f;

        for(int b = 0; b < TAM_BATCH; b++) {
            int idx = rand() % treino.numSequencias; // amostragem aleatoria, nao sequencial
            montarEntradaAlvo(treino.sequencia(idx), treino.tamJanela, entrada, alvo);

            modelo.prop(entrada);
            somaPerdaBatch += modelo.perdaCrossEntropy(alvo);
            modelo.retroprop();
        }
        otim.att();

        float perdaMediaBatch = somaPerdaBatch / (float)TAM_BATCH;

        if(passo % 20 == 0) {
            time_t agora = time(nullptr);
            printf("passo %d/%d | perda_treino=%.4f | taxa=%.6f | tempo=%lds\n",
                   passo, totalPassosBatch, perdaMediaBatch, otim.taxa, (long)(agora - inicioTreino));
        }

        if(passo % PASSOS_ENTRE_VALIDACAO == 0 && passo > 0) {
            float perdaVal = avaliarValidacao(&modelo, &validacao, 200);
            printf(">>> passo %d | perda_validacao=%.4f\n", passo, perdaVal);
            salvarCheckpoint(caminhoCheckpoint, &modelo);

            if(perdaVal < melhorPerdaValidacao) {
                melhorPerdaValidacao = perdaVal;
                salvarCheckpoint(caminhoMelhor, &modelo);
                printf(">>> novo melhor checkpoint (perda_validacao=%.4f): %s\n", perdaVal, caminhoMelhor);
            }

            modelo.defSeq(SEQ_MAX); // avaliarValidacao muda seqAtual, restaura pro batch
        }
    }

    salvarCheckpoint(caminhoCheckpoint, &modelo);
    printf("Treino concluido.\n");

    // --- 5. geracao de exemplo (usa o MELHOR checkpoint, nao necessariamente o ultimo estado) ---
    if(carregarCheckpoint(caminhoMelhor, &modelo)) {
        printf("Gerando a partir do melhor checkpoint (perda_validacao=%.4f)\n", melhorPerdaValidacao);
    } else {
        printf("Nenhum melhor checkpoint separado encontrado, gerando com o estado final.\n");
    }
    const char* promptTexto = "No princípio";
    Vetor<int> idsPrompt; idsPrompt.iniciar();
    tokenizador.codificar(promptTexto, (int)strlen(promptTexto), &idsPrompt);

    int gerados[100];
    modelo.gerarGuloso(idsPrompt.dados, idsPrompt.tam, gerados, 100);

    printf("\n=== Geracao a partir de \"%s\" ===\n", promptTexto);
    int tamTextoGerado;
    char* textoGerado = tokenizador.decodificar(gerados, 100, &tamTextoGerado);
    printf("%s%s\n", promptTexto, textoGerado);
    free(textoGerado);

    idsPrompt.liberar();
    otim.liberar();
    treino.liberar();
    validacao.liberar();
    return 0;
}
