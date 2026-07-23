// teste_modelo2.cpp
// v4: as versoes anteriores testavam so se o mecanismo (atencao/embedding/
// cross-entropy/AdamW) consegue aprender PADROES SINTETICOS simples (formula
// linear, depois tema+ruido). isso prova que o cano nao esta entupido, mas
// NAO prova que o modelo aprende linguagem natural -- que tem: distribuicao
// de frequencia extremamente desbalanceada (lei de Zipf), ambiguidade real
// (o proximo caractere nao e 100% determinado pelo contexto), e dependencias
// nao-triviais (padroes de silaba, palavras que se repetem em posicoes
// variadas).
//
// esta versao usa texto real em portugues, tokenizado char-a-char (cada
// caractere ASCII vira um id de 0-127), sem depender do tokenizador BPE do
// projeto (que nao foi enviado). isso testa a capacidade de aprendizagem
// contra a dificuldade estatistica de linguagem natural de verdade, nao
// contra um padrao artificial facil.
//
// criterio de sucesso: a perda deve cair beeem abaixo da entropia de ordem-0
// do proprio corpus (a entropia de "so saber a frequencia de cada letra,
// sem olhar o contexto"), ja que essa e a barreira real que separa "decorou
// a frequencia das letras" de "aprendeu que depois de 'qu' vem vogal" etc.
#include <cstdio>
#include <cstring>
#include <cmath>
#include "biblis/modelo.h"
#include "biblis/otimis/adamw.h"

#define VOCAB 128      // ASCII
#define SEQ 32
#define TAM_BATCH 8
#define N_EPOCAS 200
#define N_EPOCAS_MEDIA 10

// corpus pequeno mas real, em portugues, com bastante repeticao de palavras
// e estrutura (do jeito que texto de verdade tem) -- nao gerado por formula
static const char* CORPUS =
    "o gato subiu no telhado e o cachorro ficou latindo la embaixo. "
    "o gato olhou para o cachorro e continuou andando pelo telhado. "
    "no dia seguinte o cachorro subiu tambem no telhado pra brincar. "
    "o gato e o cachorro ficaram amigos e brincavam todos os dias. "
    "a menina via o gato e o cachorro brincando no telhado da casa. "
    "ela chamava o gato pelo nome e o gato descia correndo ate ela. "
    "o cachorro tambem descia latindo feliz atras da menina e do gato. "
    "todos os dias essa historia se repetia no telhado daquela casa. "
    "o gato subiu no telhado, o cachorro ficou latindo, a menina sorriu. "
    "brincar no telhado era a parte favorita do dia do gato e do cachorro. ";

// instrumentacao: norma L2 dos parametros e dos gradientes de UMA camada,
// usando a interface generica params()/gradParams() que toda Camada expoe.
// isso nao modifica nenhum header original, so consome a interface publica.
static void normasCamada(Camada* c, float* normaParam, float* normaGrad) {
    float* ptrs[16]; int tams[16];
    float* gptrs[16]; int gtams[16];
    c->params(ptrs, tams);
    c->gradParams(gptrs, gtams);
    double somaP = 0.0, somaG = 0.0;
    for(int g = 0; g < c->grupos; g++) {
        for(int i = 0; i < tams[g]; i++) somaP += (double)ptrs[g][i] * ptrs[g][i];
        for(int i = 0; i < gtams[g]; i++) somaG += (double)gptrs[g][i] * gptrs[g][i];
    }
    *normaParam = sqrtf((float)somaP);
    *normaGrad = sqrtf((float)somaG);
}

// imprime a norma de TODAS as camadas do modelo numa linha soh, rotulada
// pela epoca -- pra comparar a evolucao camada-a-camada ao longo do treino
static void imprimirNormas(Modelo* modelo, int epoca) {
    printf("  [normas epoca %d] ", epoca);
    for(int c = 0; c < modelo->totalCamadas; c++) {
        float np, ng;
        normasCamada(modelo->todasCamadas[c], &np, &ng);
        const char* nome = (c == 0) ? "emb"
                          : (c == modelo->totalCamadas - 1) ? "saida"
                          : "bloco";
        printf("%s%d[p=%.2f g=%.2f] ", nome, c, np, ng);
    }
    printf("\n");
}

int main() {
    int dim = 48;
    int nCab = 4;
    int dimFF = 192;
    int nCamadas = 3;
    int seqMax = 48;

    int tamCorpus = (int)strlen(CORPUS);
    printf("=== teste_modelo2 (capacidade real de aprender linguagem natural) ===\n");
    printf("vocab=%d(ascii) dim=%d nCab=%d dimFF=%d nCamadas=%d seqMax=%d seq=%d\n",
           VOCAB, dim, nCab, dimFF, nCamadas, seqMax, SEQ);
    printf("corpus: %d caracteres\n", tamCorpus);

    if(tamCorpus < SEQ + 1) {
        printf("corpus curto demais pra SEQ=%d\n", SEQ);
        return 1;
    }

    // --- entropia de ordem-0 do corpus: a entropia de so saber a frequencia
    // de cada caractere, ignorando contexto. isso e a barreira de verdade:
    // um modelo que so decora "que letras sao comuns" nunca fica abaixo disso ---
    int freq[VOCAB];
    memset(freq, 0, sizeof(freq));
    for(int i = 0; i < tamCorpus; i++) {
        unsigned char c = (unsigned char)CORPUS[i];
        if(c < VOCAB) freq[c]++;
    }
    float entropiaOrdem0 = 0.0f;
    for(int v = 0; v < VOCAB; v++) {
        if(freq[v] == 0) continue;
        float p = (float)freq[v] / (float)tamCorpus;
        entropiaOrdem0 += -p * std::log(p);
    }
    printf("entropia de ordem-0 (so frequencia de letras) = %f\n", entropiaOrdem0);
    printf("  -> se a perda final ficar perto ou acima disso, o modelo NAO\n");
    printf("     aprendeu nenhuma dependencia de contexto, so decorou frequencia\n");

    // --- 1. construcao ---
    Modelo modelo(VOCAB, dim, nCab, dimFF, nCamadas, seqMax);
    modelo.inicializar("xavier");
    printf("totalCamadas=%d (esperado %d)\n", modelo.totalCamadas, 1 + nCamadas + 1);

    // --- 2. otimizador ---
    AdamW otim;
    otim.iniciar(modelo.todasCamadas, modelo.totalCamadas, 3e-4f);
    printf("otimizador: totalN=%d parametros, nGrupos=%d\n", otim.totalN, otim.nGrupos);

    // --- 3. janelas de SEQ+1 caracteres, deslizando pelo corpus com passo 4
    // (janelas sobrepostas, como um dataset de treino de verdade teria) ---
    int maxInicio = tamCorpus - (SEQ + 1);
    int passoJanela = 4;
    int nJanelas = maxInicio / passoJanela + 1;
    if(nJanelas > 200) nJanelas = 200;

    int (*janelas)[SEQ + 1] = new int[nJanelas][SEQ + 1];
    for(int j = 0; j < nJanelas; j++) {
        int inicio = j * passoJanela;
        if(inicio > maxInicio) inicio = maxInicio;
        for(int i = 0; i < SEQ + 1; i++) {
            janelas[j][i] = (unsigned char)CORPUS[inicio + i];
        }
    }
    printf("nJanelas=%d (passo=%d)\n", nJanelas, passoJanela);

    int entrada[SEQ], alvo[SEQ];
    modelo.defSeq(SEQ);

    // --- 4. treino por epoca, batch real ---
    float historicoEpoca[N_EPOCAS];

    for(int epoca = 0; epoca < N_EPOCAS; epoca++) {
        float somaPerdaEpoca = 0.0f;
        int contPerdas = 0;

        for(int inicio = 0; inicio < nJanelas; inicio += TAM_BATCH) {
            int fimBatch = inicio + TAM_BATCH;
            if(fimBatch > nJanelas) fimBatch = nJanelas;

            modelo.zerarGrad();
            float somaPerdaBatch = 0.0f;

            for(int j = inicio; j < fimBatch; j++) {
                for(int i = 0; i < SEQ; i++) {
                    entrada[i] = janelas[j][i];
                    alvo[i] = janelas[j][i + 1];
                }
                modelo.prop(entrada);
                somaPerdaBatch += modelo.perdaCrossEntropy(alvo);
                modelo.retroprop();
                contPerdas++;
            }
            otim.att();
            somaPerdaEpoca += somaPerdaBatch;

            if(!std::isfinite(somaPerdaBatch)) {
                printf("\nRESULTADO: FALHOU (perda nao finita na epoca %d)\n", epoca);
                delete[] janelas;
                otim.liberar();
                return 1;
            }
        }

        historicoEpoca[epoca] = somaPerdaEpoca / (float)contPerdas;
        if(epoca % 10 == 0 || epoca == N_EPOCAS - 1) {
            printf("epoca %d: perda_media = %f\n", epoca, historicoEpoca[epoca]);
            imprimirNormas(&modelo, epoca);
        }
    }

    // --- 5. checagens contra a entropia de ordem-0 real do corpus ---
    float mediaFim = 0.0f;
    for(int i = 0; i < N_EPOCAS_MEDIA; i++) mediaFim += historicoEpoca[N_EPOCAS - 1 - i];
    mediaFim /= N_EPOCAS_MEDIA;

    float variancia = 0.0f;
    for(int i = 0; i < N_EPOCAS_MEDIA; i++) {
        float d = historicoEpoca[N_EPOCAS - 1 - i] - mediaFim;
        variancia += d * d;
    }
    variancia /= N_EPOCAS_MEDIA;
    float desvio = std::sqrt(variancia);

    bool aprendeuContexto = mediaFim < entropiaOrdem0 * 0.7f;
    bool travouNaMarginal = mediaFim > entropiaOrdem0 * 0.9f;

    // geracao real: continua "o gato subiu no telhado e o " e olha se sai
    // algo pareceido com texto (nao necessariamente perfeito, mas nao deve
    // ser lixo puro nem colapsar num caractere so)
    const char* prompt = "o gato subiu no telhado e o ";
    int tamPrompt = (int)strlen(prompt);
    int idsPrompt[64];
    for(int i = 0; i < tamPrompt; i++) idsPrompt[i] = (unsigned char)prompt[i];

    int gerados[40];
    modelo.gerarGuloso(idsPrompt, tamPrompt, gerados, 40);

    char textoGerado[41];
    for(int i = 0; i < 40; i++) {
        int c = gerados[i];
        textoGerado[i] = (c >= 32 && c < 127) ? (char)c : '?';
    }
    textoGerado[40] = '\0';

    printf("\nprompt: \"%s\"\n", prompt);
    printf("gerado: \"%s\"\n", textoGerado);

    bool semColapsoRepeticao = false;
    for(int i = 1; i < 40; i++) {
        if(gerados[i] != gerados[0]) { semColapsoRepeticao = true; break; }
    }
    // checagem mais forte: nao pode ter mais de 6 caracteres identicos seguidos
    bool semRepeticaoLonga = true;
    int seguidos = 1;
    for(int i = 1; i < 40; i++) {
        if(gerados[i] == gerados[i-1]) {
            seguidos++;
            if(seguidos > 6) { semRepeticaoLonga = false; break; }
        } else {
            seguidos = 1;
        }
    }

    printf("\nperda final (media ult %d epocas) = %f\n", N_EPOCAS_MEDIA, mediaFim);
    printf("desvio = %f\n", desvio);
    printf("entropia ordem-0 = %f | 70%% disso = %f | 90%% disso = %f\n",
           entropiaOrdem0, entropiaOrdem0 * 0.7f, entropiaOrdem0 * 0.9f);

    printf("\naprendeu dependencia de contexto (perda < 70%% da entropia ordem-0)? %s\n",
           aprendeuContexto ? "sim (OK)" : "nao (FALHOU)");
    if(travouNaMarginal) {
        printf(">>> SINTOMA DETECTADO: perda travou perto da entropia de ordem-0\n");
        printf(">>> (modelo aprendeu so a frequencia das letras, nao contexto real)\n");
    }
    printf("geracao sem colapso total em 1 caractere? %s\n",
           semColapsoRepeticao ? "sim (OK)" : "nao (FALHOU)");
    printf("geracao sem repeticao longa (>6 seguidos)? %s\n",
           semRepeticaoLonga ? "sim (OK)" : "nao (FALHOU)");

    bool ok = aprendeuContexto && semColapsoRepeticao && semRepeticaoLonga;
    printf("\nRESULTADO: %s\n", ok ? "PASSOU" : "FALHOU");

    delete[] janelas;
    otim.liberar();
    return ok ? 0 : 1;
}
