// chat.cpp
// Treina um modelo do zero usando SO biblia.txt (BPE treinado do zero nela),
// e depois abre um loop de conversa interativo no terminal.
//
// Diferenca em relacao a treino.cpp: aqui o treino e sempre do-zero, com
// corpus fixo (biblia.txt) e BPE proprio, e ao final NAO encerra: fica
// esperando voce digitar mensagens e responde usando geracao gulosa.
//
// Uso: ./chat biblia.txt
// (gera merges_biblia.txt, vocab_biblia.txt, treino_biblia.bin,
//  validacao_biblia.bin, checkpoint_biblia.bin, tudo no diretorio atual)
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <ctime>
#include <string>
#include "biblis/modelo.h"
#include "biblis/otimis/adamw.h"
#include "biblis/toke/bpe.h"
#include "biblis/toke/fabrica_dados.h"

#define TAM_BATCH 8
#define DIM 128
#define N_CAB 4
#define DIM_FF 512
#define N_CAMADAS 4
#define SEQ_MAX 64
#define TAXA_MAX 3e-4f
#define TAXA_MIN 3e-5f
#define PASSOS_AQUECIMENTO 100
#define PASSOS_TREINO 1000
#define TAM_RESPOSTA 80
#define MAX_MERGES 4000

// ---------------------------------------------------------------------------
// leitura dos binarios de dados de treino/validacao (identico a treino.cpp)
// ---------------------------------------------------------------------------
struct DadosTreino {
    int32_t tamVocab;
    int32_t tamJanela;
    int32_t numSequencias;
    int32_t* dados;

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
            printf("Numero magico invalido em %s\n", caminho);
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

    const int32_t* sequencia(int i) const {
        return dados + (size_t)i * tamJanela;
    }

    void liberar() { free(dados); }
};

static void montarEntradaAlvo(const int32_t* janela, int tamJanela, int* entrada, int* alvo) {
    int seq = tamJanela - 1;
    for(int i = 0; i < seq; i++) {
        entrada[i] = (int)janela[i];
        alvo[i] = (int)janela[i + 1];
    }
}

// ---------------------------------------------------------------------------
// gera uma resposta a partir de um prompt do usuario, usando geracao gulosa
// ---------------------------------------------------------------------------
static void gerarResposta(Modelo* modelo, TokenizadorBPE* tokenizador, const char* prompt) {
    Vetor<int> idsPrompt; idsPrompt.iniciar();
    tokenizador->codificar(prompt, (int)strlen(prompt), &idsPrompt);

    if(idsPrompt.tam == 0) {
        printf("(nao entendi, tente outra frase)\n");
        idsPrompt.liberar();
        return;
    }

    int* gerados = (int*)malloc(TAM_RESPOSTA * sizeof(int));
    modelo->gerarGuloso(idsPrompt.dados, idsPrompt.tam, gerados, TAM_RESPOSTA);

    int tamTexto;
    char* texto = tokenizador->decodificar(gerados, TAM_RESPOSTA, &tamTexto);
    printf("IA: %s\n", texto);

    free(texto);
    free(gerados);
    idsPrompt.liberar();

    modelo->defSeq(SEQ_MAX); // gerarGuloso mudou seqAtual, restaura pro que vier depois
}

int main(int argc, char** argv) {
    if(argc < 2) {
        printf("Uso: %s biblia.txt\n", argv[0]);
        return 1;
    }
    const char* caminhoCorpus = argv[1];

    // checagem de sanidade: o corpus precisa existir e ser legivel
    {
        FILE* teste = fopen(caminhoCorpus, "rb");
        if(!teste) {
            printf("ERRO: nao consegui abrir '%s'. Confira o caminho.\n", caminhoCorpus);
            return 1;
        }
        fclose(teste);
    }

    const char* caminhoMerges     = "merges_biblia.txt";
    const char* caminhoVocab      = "vocab_biblia.txt";
    const char* caminhoTreinoBin  = "treino_biblia.bin";
    const char* caminhoValBin     = "validacao_biblia.bin";
    const char* caminhoCheckpoint = "checkpoint_biblia.bin";

    // --- 1. treinar BPE do zero, so no corpus dado ---
    printf("=== Treinando BPE do zero em %s ===\n", caminhoCorpus);
    FabricaDados::treinarEsalvar(caminhoCorpus, caminhoMerges, caminhoVocab, MAX_MERGES);

    // --- 2. gerar .bin de treino/validacao ---
    TokenizadorBPE tokenizador;
    FabricaDados::carregarTokenizador(&tokenizador, caminhoMerges, caminhoVocab);
    FabricaDados::gerar(caminhoCorpus, &tokenizador, caminhoTreinoBin, caminhoValBin);

    // --- 3. carregar dados ---
    DadosTreino treino, validacao;
    if(!treino.carregar(caminhoTreinoBin)) return 1;
    if(!validacao.carregar(caminhoValBin)) return 1;

    if(treino.tamJanela != SEQ_MAX + 1) {
        printf("tamJanela (%d) nao bate com SEQ_MAX+1 (%d)\n", treino.tamJanela, SEQ_MAX + 1);
        return 1;
    }

    // --- 4. modelo + otimizador ---
    Modelo modelo(treino.tamVocab, DIM, N_CAB, DIM_FF, N_CAMADAS, SEQ_MAX);
    modelo.inicializar("xavier");

    AdamW otim;
    otim.iniciar(modelo.todasCamadas, modelo.totalCamadas, TAXA_MAX);
    printf("Modelo: %d parametros, %d grupos\n", otim.totalN, otim.nGrupos);

    // --- 5. loop de treino (identico em espirito ao treino.cpp) ---
    modelo.defSeq(SEQ_MAX);
    int entrada[SEQ_MAX], alvo[SEQ_MAX];

    int totalPassosBatch = treino.numSequencias / TAM_BATCH;
    if(PASSOS_TREINO < totalPassosBatch) totalPassosBatch = PASSOS_TREINO;
    if(totalPassosBatch < 1) totalPassosBatch = 1;
    printf("Iniciando treino: %d sequencias, batch=%d, %d passos\n",
           treino.numSequencias, TAM_BATCH, totalPassosBatch);

    AgendadorCosseno agendador;
    agendador.taxaMax = TAXA_MAX;
    agendador.taxaMin = TAXA_MIN;
    agendador.passosTotal = totalPassosBatch;
    agendador.aquecimento = PASSOS_AQUECIMENTO < totalPassosBatch ? PASSOS_AQUECIMENTO : totalPassosBatch / 10;

    srand((unsigned int)time(nullptr));
    time_t inicioTreino = time(nullptr);

    for(int passo = 0; passo < totalPassosBatch; passo++) {
        otim.taxa = agendador.calcular(passo);

        modelo.zerarGrad();
        float somaPerdaBatch = 0.0f;

        for(int b = 0; b < TAM_BATCH; b++) {
            int idx = rand() % treino.numSequencias;
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
    }

    // salva checkpoint final, so por seguranca (nao e recarregado neste programa)
    {
        FILE* a = fopen(caminhoCheckpoint, "wb");
        if(a) {
            for(int c = 0; c < modelo.totalCamadas; c++) {
                float* ptrs[16]; int tams[16];
                modelo.todasCamadas[c]->params(ptrs, tams);
                for(int g = 0; g < modelo.todasCamadas[c]->grupos; g++) {
                    fwrite(ptrs[g], sizeof(float), tams[g], a);
                }
            }
            fclose(a);
            printf("Checkpoint salvo: %s\n", caminhoCheckpoint);
        }
    }

    printf("\nTreino concluido. Agora voce pode conversar (digite 'sair' para encerrar).\n\n");

    // --- 6. loop de chat interativo ---
    char linha[2048];
    while(true) {
        printf("Voce: ");
        fflush(stdout);
        if(!fgets(linha, sizeof(linha), stdin)) break;

        int tam = (int)strlen(linha);
        while(tam > 0 && (linha[tam-1] == '\n' || linha[tam-1] == '\r')) linha[--tam] = '\0';

        if(tam == 0) continue;
        if(strcmp(linha, "sair") == 0 || strcmp(linha, "exit") == 0) break;

        gerarResposta(&modelo, &tokenizador, linha);
    }

    printf("Encerrando.\n");

    otim.liberar();
    treino.liberar();
    validacao.liberar();
    return 0;
}
