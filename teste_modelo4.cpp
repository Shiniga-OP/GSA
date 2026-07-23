// teste_pipeline.cpp
// teste de pipeline completo: BPE -> Modelo -> treino AdamW -> GERACAO DE TEXTO
// objetivo: ver o texto gerado em varios pontos do treino, nao so os numeros.
// compilar na pasta mae, onde a pasta biblis/ fica ao lado deste arquivo:
//   g++ -std=c++17 -O2 -o teste_pipeline teste_pipeline.cpp
#include <stdio.h>
#include <math.h>
#include "biblis/util.h"
#include "biblis/toke/bpe.h"
#include "biblis/toke/fabrica_dados.h"
#include "biblis/modelo.h"
#include "biblis/otimis/otimizador.h"
#include "biblis/otimis/adamw.h"

// sampling por temperatura sobre os ultimos logits (evita loop travado do
// gerarGuloso puro). nao mexe na biblioteca, so le modelo.logits por fora.
static int amostrarTemperatura(const float* logits, int vocab, float temperatura) {
    float mx = logits[0];
    for(int v = 1; v < vocab; v++) if(logits[v] > mx) mx = logits[v];

    float* probs = (float*)malloc(vocab * sizeof(float));
    float soma = 0.0f;
    for(int v = 0; v < vocab; v++) {
        probs[v] = expf((logits[v] - mx) / temperatura);
        soma += probs[v];
    }
    float alvo = ((float)rand() / (float)RAND_MAX) * soma;
    float acum = 0.0f;
    int escolhido = vocab - 1;
    for(int v = 0; v < vocab; v++) {
        acum += probs[v];
        if(acum >= alvo) { escolhido = v; break; }
    }
    free(probs);
    return escolhido;
}

// gera texto usando o modelo em qualquer ponto do treino, com temperatura
static void gerarTexto(Modelo& modelo, TokenizadorBPE& tok, const int* semente,
int tamSemente, int tamGerar, float temperatura, char* saidaTexto, int capSaida) {
    int seqMax = modelo.seqMax;
    int* buf = (int*)malloc(seqMax * sizeof(int));
    int tamBuf = tamSemente < seqMax ? tamSemente : seqMax;
    memcpy(buf, semente + (tamSemente - tamBuf), tamBuf * sizeof(int));

    int* todosIds = (int*)malloc((tamSemente + tamGerar) * sizeof(int));
    memcpy(todosIds, semente, tamSemente * sizeof(int));

    for(int g = 0; g < tamGerar; g++) {
        modelo.defSeq(tamBuf);
        modelo.prop(buf);

        float* ultimoLogit = modelo.logits + (tamBuf - 1) * modelo.vocab;
        int escolhido = amostrarTemperatura(ultimoLogit, modelo.vocab, temperatura);
        todosIds[tamSemente + g] = escolhido;

        if(tamBuf < seqMax) {
            buf[tamBuf] = escolhido;
            tamBuf++;
        } else {
            memmove(buf, buf + 1, (seqMax - 1) * sizeof(int));
            buf[seqMax - 1] = escolhido;
        }
    }

    int tamDec;
    char* decodificado = tok.decodificar(todosIds, tamSemente + tamGerar, &tamDec);
    int copiar = tamDec < capSaida - 1 ? tamDec : capSaida - 1;
    memcpy(saidaTexto, decodificado, copiar);
    saidaTexto[copiar] = '\0';
    free(decodificado);
    free(todosIds);
    free(buf);
}

// === checkpoint: salva/carrega pesos de todas as camadas + estado do AdamW ===
// a biblioteca nao tem persistencia embutida; isso le os ponteiros de params()
// de cada camada (mesmo mecanismo que o otimizador usa) e grava em binario.
// cabecalho com metadados evita a causa do segfault anterior: carregar um
// checkpoint de um modelo com outra forma (dim/vocab/camadas diferentes)
// sobrescrevia os pesos parcialmente antes de detectar a incompatibilidade.
struct CabecalhoCheckpoint {
    int32_t numeroMagico; // 0x4B434B42 = "BKCK" em little-endian
    int32_t vocab;
    int32_t dim;
    int32_t nCab;
    int32_t dimFF;
    int32_t nCamadas;
    int32_t seqMax;
    int32_t passoOtim;
    int32_t totalParamsOtim; // otim.totalN, valida o tamanho do estado AdamW
};
#define CHECKPOINT_NUMERO_MAGICO 0x4B434B42

static void salvarCheckpoint(const char* caminho, Modelo& modelo, AdamW& otim) {
    FILE* a = fopen(caminho, "wb");
    if(!a) { printf("ERRO: nao consegui salvar checkpoint em %s\n", caminho); return; }

    CabecalhoCheckpoint cab;
    cab.numeroMagico = CHECKPOINT_NUMERO_MAGICO;
    cab.vocab = modelo.vocab;
    cab.dim = modelo.dim;
    cab.nCab = -1; // multicabeca nao expoe nCab em Modelo; validado via totalParamsOtim
    cab.dimFF = modelo.dimFF;
    cab.nCamadas = modelo.nCamadas;
    cab.seqMax = modelo.seqMax;
    cab.passoOtim = otim.passo;
    cab.totalParamsOtim = otim.totalN;
    fwrite(&cab, sizeof(CabecalhoCheckpoint), 1, a);

    for(int c = 0; c < modelo.totalCamadas; c++) {
        float* ptrs[16]; int tams[16];
        modelo.todasCamadas[c]->params(ptrs, tams);
        for(int g = 0; g < modelo.todasCamadas[c]->grupos; g++) {
            fwrite(ptrs[g], sizeof(float), tams[g], a);
        }
    }
    fwrite(otim.estado, sizeof(float), 2 * otim.totalN, a);

    fclose(a);
    printf("checkpoint salvo: %s (passo=%d)\n", caminho, cab.passoOtim);
}

// le tudo para buffers temporarios primeiro; so escreve nos pesos reais do
// modelo se o arquivo inteiro for lido com sucesso E o cabecalho bater.
// isso evita deixar o modelo com pesos parcialmente sobrescritos em caso
// de incompatibilidade ou arquivo truncado (causa do segfault anterior).
static bool carregarCheckpoint(const char* caminho, Modelo& modelo, AdamW& otim) {
    FILE* a = fopen(caminho, "rb");
    if(!a) return false;

    CabecalhoCheckpoint cab;
    if(fread(&cab, sizeof(CabecalhoCheckpoint), 1, a) != 1) {
        fclose(a);
        printf("checkpoint ignorado: arquivo menor que o cabecalho esperado\n");
        return false;
    }
    if(cab.numeroMagico != CHECKPOINT_NUMERO_MAGICO) {
        fclose(a);
        printf("checkpoint ignorado: numero magico nao bate (arquivo de formato antigo/diferente)\n");
        return false;
    }
    if(cab.vocab != modelo.vocab || cab.dim != modelo.dim ||
    cab.dimFF != modelo.dimFF || cab.nCamadas != modelo.nCamadas ||
    cab.seqMax != modelo.seqMax || cab.totalParamsOtim != otim.totalN) {
        fclose(a);
        printf("checkpoint ignorado: forma do modelo mudou (vocab=%d dim=%d dimFF=%d "
        "nCamadas=%d seqMax=%d totalParams=%d), nao bate com o checkpoint salvo\n",
        modelo.vocab, modelo.dim, modelo.dimFF, modelo.nCamadas, modelo.seqMax, otim.totalN);
        return false;
    }

    // buffers temporarios: le tudo antes de tocar nos pesos reais
    int totalPesos = 0;
    for(int c = 0; c < modelo.totalCamadas; c++) totalPesos += modelo.todasCamadas[c]->numParams();

    float* pesosTmp = (float*)malloc(totalPesos * sizeof(float));
    float* estadoTmp = (float*)malloc(2 * otim.totalN * sizeof(float));
    if(!pesosTmp || !estadoTmp) {
        free(pesosTmp); free(estadoTmp); fclose(a);
        printf("ERRO: falha ao alocar buffers temporarios pro checkpoint\n");
        return false;
    }

    size_t lidoPesos = fread(pesosTmp, sizeof(float), totalPesos, a);
    size_t lidoEstado = fread(estadoTmp, sizeof(float), 2 * otim.totalN, a);
    fclose(a);

    if((int)lidoPesos != totalPesos || (int)lidoEstado != 2 * otim.totalN) {
        free(pesosTmp); free(estadoTmp);
        printf("ERRO: checkpoint truncado (dados incompletos). modelo NAO foi alterado.\n");
        return false;
    }

    // tudo lido com sucesso: agora sim aplica nos pesos reais
    int pos = 0;
    for(int c = 0; c < modelo.totalCamadas; c++) {
        float* ptrs[16]; int tams[16];
        modelo.todasCamadas[c]->params(ptrs, tams);
        for(int g = 0; g < modelo.todasCamadas[c]->grupos; g++) {
            memcpy(ptrs[g], pesosTmp + pos, tams[g] * sizeof(float));
            pos += tams[g];
        }
    }
    memcpy(otim.estado, estadoTmp, 2 * otim.totalN * sizeof(float));
    otim.passo = cab.passoOtim;

    free(pesosTmp);
    free(estadoTmp);
    printf("checkpoint carregado: %s (retomando do passo=%d)\n", caminho, cab.passoOtim);
    return true;
}

int main() {
    srand(42);

    // === 1. treino BPE ===
    int tamTexto;
    char* texto = FabricaDados::lerArquivoTexto("biblia.txt", &tamTexto);
    if(!texto) {
        printf("FALHOU: nao leu corpus\n");
        return 1;
    }
    printf("corpus lido: %d bytes\n", tamTexto);

    TreinadorBPE treinador;
    treinador.treinar(texto, tamTexto, 4000);
    treinador.salvar("merges_teste.txt");

    TokenizadorBPE tok;
    tok.carregarMerges("merges_teste.txt");
    tok.construirVocab(texto, tamTexto);

    int vocab = tok.vocabTam();
    printf("vocab: %d tokens\n", vocab);

    Vetor<int> tokens; tokens.iniciar();
    tok.codificar(texto, tamTexto, &tokens);
    printf("tokens codificados: %d\n", tokens.tam);
    free(texto);

    if(tokens.tam < 32) {
        printf("FALHOU: poucos tokens pra montar sequencias de treino\n");
        return 1;
    }

    // === 2. modelo em escala maior, condizente com corpus de ~4MB ===
    int dim = 256;
    int nCab = 8;
    int dimFF = 1024;
    int nCamadas = 6;
    int seqMax = 128;

    Modelo modelo(vocab, dim, nCab, dimFF, nCamadas, seqMax);
    modelo.inicializar("xavier");
    printf("modelo criado: vocab=%d dim=%d nCab=%d dimFF=%d nCamadas=%d seqMax=%d\n",
        vocab, dim, nCab, dimFF, nCamadas, seqMax);

    AdamW otim;
    otim.iniciar(modelo.todasCamadas, modelo.totalCamadas, 3e-3f);

    int passoInicial = 0;
    if(carregarCheckpoint("checkpoint.bin", modelo, otim)) {
        passoInicial = otim.passo;
    }

    // === 3. sequencias de treino: janelas deslizantes de seqMax+1 sobre os tokens ===
    int seq = seqMax;
    int stride = 64; // usado tambem no calculo de nSeqs abaixo; deve bater com o loop
    int nSeqs = 0;
    for(int pos = 0; pos + seq + 1 <= tokens.tam; pos += stride) nSeqs++;
    if(nSeqs == 0) {
        printf("FALHOU: nenhuma sequencia de treino formada\n");
        return 1;
    }
    printf("sequencias de treino: %d\n", nSeqs);

    int* idsEnt = (int*)malloc(seq * sizeof(int));
    int* idsAlvo = (int*)malloc(seq * sizeof(int));

    int tamSemente = 5;
    int* semente = (int*)malloc(tamSemente * sizeof(int));
    for(int i = 0; i < tamSemente; i++) semente[i] = tokens[i];

    char bufTexto[2048];

    // === 4. loop de treino com AgendadorCosseno (warmup + decay cosseno) ===
    // ESTE E UM TESTE DE VALIDACAO DE ESCALA, nao o treino final: poucos passos,
    // so pra confirmar que o corpus grande + modelo maior nao trava, nao estoura
    // memoria, e a perda cai de forma estavel. o treino de verdade roda por muito
    // mais passos, retomando deste mesmo checkpoint.bin.
    int passos = 2000;
    int aquecimento = 100;
    bool viuNan = false;

    AgendadorCosseno agenda;
    agenda.taxaMax = 3e-4f;
    agenda.taxaMin = 1e-6f;
    agenda.passosTotal = passos;
    agenda.aquecimento = aquecimento;

    modelo.defSeq(seq);

    for(int p = passoInicial; p < passos; p++) {
        int base = (p % nSeqs) * stride;
        for(int i = 0; i < seq; i++) {
            idsEnt[i]  = tokens[base + i];
            idsAlvo[i] = tokens[base + i + 1];
        }

        otim.taxa = agenda.calcular(p);

        modelo.defSeq(seq);
        modelo.zerarGrad();
        modelo.prop(idsEnt);
        float perda = modelo.perdaCrossEntropy(idsAlvo);
        modelo.retroprop();
        otim.att();

        if(isnan(perda) || isinf(perda)) {
            printf("FALHOU no passo %d: perda = %f (NaN/Inf)\n", p, perda);
            viuNan = true;
            break;
        }

        // gera texto e salva checkpoint periodicamente
        if(p % 100 == 0 || p == passos - 1) {
            gerarTexto(modelo, tok, semente, tamSemente, 40, 0.8f, bufTexto, sizeof(bufTexto));
            printf("\n--- passo %d (perda=%.4f) ---\n%s\n", p, perda, bufTexto);
            salvarCheckpoint("checkpoint.bin", modelo, otim);
        }
    }

    free(idsEnt);
    free(idsAlvo);

    printf("\n=== STATUS ===\n");
    if(viuNan) {
        printf("FALHOU (NaN/Inf durante o treino)\n");
    } else {
        printf("treino concluido sem NaN/Inf. Avalie o texto gerado acima:\n");
        printf("- faz frases com sentido em portugues (sujeito+verbo+objeto)?\n");
        printf("- reaproveita palavras do corpus de forma coerente, nao so copia trechos inteiros?\n");
        printf("- varia entre as chamadas (nao trava repetindo o mesmo token)?\n");
    }

    // === 5. geracao final com temperaturas diferentes, pra comparar ===
    if(!viuNan) {
        printf("\n=== GERACAO FINAL, temperaturas diferentes ===\n");
        for(int t = 0; t < 3; t++) {
            float temp = (t == 0) ? 0.5f : (t == 1) ? 0.9f : 1.3f;
            gerarTexto(modelo, tok, semente, tamSemente, 60, temp, bufTexto, sizeof(bufTexto));
            printf("\n[temperatura=%.1f]\n%s\n", temp, bufTexto);
        }
    }

    free(semente);
    otim.liberar();
    tokens.liberar();
    return (viuNan) ? 1 : 0;
}
