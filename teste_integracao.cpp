// teste_integracao_minimo.cpp
// Teste de PONTA A PONTA, sem nenhum isolamento/mock: usa o pipeline REAL
// (TreinadorBPE + TokenizadorBPE + FabricaDados + Modelo + AdamW) exatamente
// como treino.cpp usa, mas com um corpus sintetico minusculo e 100% previsivel.
//
// Ideia: se o pipeline inteiro estiver certo, um modelo pequeno treinado
// numa sequencia ciclica simples ("A B C A B C A B C ...") DEVE conseguir
// prever o proximo token com folga acima do chute por frequencia (unigrama),
// porque aqui o padrao e trivial (repeticao de periodo 3).
//
// Criterio de sucesso definido ANTES de rodar:
//   - perda_unigrama_teorica = entropia da distribuicao de frequencia dos
//     tokens do corpus (chute cego, sem olhar contexto)
//   - o modelo DEVE terminar com perda de treino bem abaixo disso
//   - a geracao gulosa a partir de "A B C A B C" deve continuar "A B C A B C..."
//     (nao pode travar num unico token repetido)
//
// Compilar (a partir da pasta que contem biblis/):
//   g++ -O2 -std=c++17 -I. teste_integracao_minimo.cpp -o teste_integracao_minimo
// Rodar:
//   ./teste_integracao_minimo
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <string>
#include <map>
#include "biblis/modelo.h"
#include "biblis/otimis/adamw.h"
#include "biblis/toke/bpe.h"
#include "biblis/toke/fabrica_dados.h"

#define DIM 32
#define N_CAB 2
#define DIM_FF 64
#define N_CAMADAS 2
#define SEQ_MAX 64
#define TAM_BATCH 4
#define TAXA 3e-4f
#define PASSOS 300

static void montarEntradaAlvo(const int32_t* janela, int tamJanela, int* entrada, int* alvo) {
    int seq = tamJanela - 1;
    for(int i = 0; i < seq; i++) {
        entrada[i] = (int)janela[i];
        alvo[i] = (int)janela[i + 1];
    }
}

int main() {
    srand(42);
    printf("=== teste_integracao_minimo (pipeline real completo) ===\n\n");

    // -----------------------------------------------------------------
    // 1. corpus sintetico: padrao ciclico trivial, repetido bastante
    //    pra dar sequencias suficientes pro fabrica_dados fatiar.
    // -----------------------------------------------------------------
    const char* palavras[3] = {"alfa", "beta", "gama"};
    std::string corpus;
    for(int i = 0; i < 2000; i++) {
        corpus += palavras[i % 3];
        corpus += " ";
    }
    const char* caminhoCorpus = "corpus_teste.txt";
    {
        FILE* f = fopen(caminhoCorpus, "wb");
        if(!f) { printf("ERRO: nao consegui abrir '%s' para escrita (checar permissao/diretorio)\n", caminhoCorpus); return 1; }
        fwrite(corpus.data(), 1, corpus.size(), f);
        fclose(f);
    }
    printf("Corpus sintetico gerado: %zu chars, padrao 'alfa beta gama' repetido\n", corpus.size());

    // teste de sanidade do proprio ambiente: garante que da pra escrever
    // e reabrir um arquivo aqui, ANTES de chamar qualquer coisa do FabricaDados
    {
        FILE* f = fopen(caminhoCorpus, "rb");
        if(!f) {
            printf("ERRO: escrevi '%s' mas nao consegui reabrir para leitura. Abortando.\n", caminhoCorpus);
            return 1;
        }
        fclose(f);
    }

    // -----------------------------------------------------------------
    // 2. treinar BPE de verdade e salvar merges/vocab (pipeline real)
    // -----------------------------------------------------------------
    const char* caminhoMerges = "merges_teste.txt";
    const char* caminhoVocab  = "vocab_teste.txt";
    FabricaDados::treinarEsalvar(caminhoCorpus, caminhoMerges, caminhoVocab, 50);

    // -----------------------------------------------------------------
    // 3. gerar .bin de treino/validacao de verdade via FabricaDados
    // -----------------------------------------------------------------
    TokenizadorBPE tokParaGerar;
    FabricaDados::carregarTokenizador(&tokParaGerar, caminhoMerges, caminhoVocab);

    const char* caminhoTreinoBin = "treino_teste.bin";
    const char* caminhoValBin    = "val_teste.bin";
    FabricaDados::gerar(caminhoCorpus, &tokParaGerar, caminhoTreinoBin, caminhoValBin);

    // -----------------------------------------------------------------
    // 4. carregar os .bin gerados (igual treino.cpp faz)
    // -----------------------------------------------------------------
    struct DadosTreino {
        int32_t tamVocab, tamJanela, numSequencias;
        int32_t* dados;
        bool carregar(const char* caminho) {
            FILE* a = fopen(caminho, "rb");
            if(!a) { printf("Erro ao abrir %s\n", caminho); return false; }
            CabecalhoDadosTreino cab;
            if(fread(&cab, sizeof(cab), 1, a) != 1) { fclose(a); return false; }
            if(cab.numeroMagico != FABRICA_NUMERO_MAGICO) { fclose(a); return false; }
            tamVocab = cab.tamVocab; tamJanela = cab.tamJanela; numSequencias = cab.numSequencias;
            int totalInts = numSequencias * tamJanela;
            dados = (int32_t*)malloc(totalInts * sizeof(int32_t));
            int lidos = (int)fread(dados, sizeof(int32_t), totalInts, a);
            fclose(a);
            if(lidos != totalInts) { free(dados); return false; }
            return true;
        }
        const int32_t* sequencia(int i) const { return dados + (size_t)i * tamJanela; }
    };

    DadosTreino treino;
    if(!treino.carregar(caminhoTreinoBin)) { printf("FALHA ao carregar .bin de treino\n"); return 1; }
    printf("Treino carregado: %d sequencias, janela=%d, vocab=%d\n",
           treino.numSequencias, treino.tamJanela, treino.tamVocab);

    if(treino.tamJanela - 1 > SEQ_MAX) {
        printf("ERRO: tamJanela-1 (%d) > SEQ_MAX (%d) deste teste. Ajuste SEQ_MAX.\n", treino.tamJanela-1, SEQ_MAX);
        return 1;
    }

    // -----------------------------------------------------------------
    // 5. checagem MANUAL de sanidade: decodificar sequencia 0 (entrada/alvo)
    //    e imprimir, pra ver a olho nu se o alinhamento faz sentido
    // -----------------------------------------------------------------
    {
        int seqLen = treino.tamJanela - 1;
        int entrada[SEQ_MAX], alvo[SEQ_MAX];
        montarEntradaAlvo(treino.sequencia(0), treino.tamJanela, entrada, alvo);
        int tamTxt;
        char* txtEntrada = tokParaGerar.decodificar(entrada, seqLen, &tamTxt);
        char* txtAlvo    = tokParaGerar.decodificar(alvo, seqLen, &tamTxt);
        printf("\n[Sanidade] sequencia 0 decodificada:\n");
        printf("  entrada: \"%s\"\n", txtEntrada);
        printf("  alvo   : \"%s\"  (deve ser a entrada deslocada em 1 token)\n\n", txtAlvo);
        free(txtEntrada);
        free(txtAlvo);
    }

    // -----------------------------------------------------------------
    // 6. calcular perda de unigrama teorica (chute por frequencia) sobre
    //    o proprio corpus de treino, pra ter um "piso" de comparacao
    // -----------------------------------------------------------------
    std::map<int,int> freq;
    int totalTokensAlvo = 0;
    for(int i = 0; i < treino.numSequencias; i++) {
        int entrada[SEQ_MAX], alvo[SEQ_MAX];
        montarEntradaAlvo(treino.sequencia(i), treino.tamJanela, entrada, alvo);
        for(int t = 0; t < treino.tamJanela - 1; t++) { freq[alvo[t]]++; totalTokensAlvo++; }
    }
    float perdaUnigrama = 0.0f;
    for(auto& kv : freq) {
        float p = (float)kv.second / (float)totalTokensAlvo;
        perdaUnigrama += -p * logf(p) * ((float)kv.second); // soma ponderada de -log(p) por ocorrencia
    }
    perdaUnigrama /= (float)totalTokensAlvo;
    printf("[Piso] perda_unigrama teorica (chute por frequencia, SEM contexto) = %.4f\n\n", perdaUnigrama);

    // -----------------------------------------------------------------
    // 7. treinar o Modelo real, pequeno, poucos passos
    // -----------------------------------------------------------------
    Modelo modelo(treino.tamVocab, DIM, N_CAB, DIM_FF, N_CAMADAS, SEQ_MAX);
    modelo.inicializar("xavier");

    AdamW otim;
    otim.iniciar(modelo.todasCamadas, modelo.totalCamadas, TAXA);

    modelo.defSeq(SEQ_MAX);
    int entrada[SEQ_MAX], alvo[SEQ_MAX];
    int seqLen = treino.tamJanela - 1;

    float perdaFinal = -1.0f;
    for(int passo = 0; passo < PASSOS; passo++) {
        modelo.zerarGrad();
        float somaPerda = 0.0f;
        for(int b = 0; b < TAM_BATCH; b++) {
            int idx = rand() % treino.numSequencias;
            montarEntradaAlvo(treino.sequencia(idx), treino.tamJanela, entrada, alvo);
            modelo.prop(entrada);
            somaPerda += modelo.perdaCrossEntropy(alvo);
            modelo.retroprop();
        }
        otim.att();
        perdaFinal = somaPerda / TAM_BATCH;
        if(passo % 50 == 0 || passo == PASSOS - 1) {
            printf("passo %d/%d | perda=%.4f\n", passo, PASSOS, perdaFinal);
        }
    }

    printf("\n[Resultado] perda final de treino = %.4f\n", perdaFinal);
    printf("[Resultado] piso de unigrama       = %.4f\n", perdaUnigrama);
    bool perdaAbaixoDoUnigrama = perdaFinal < perdaUnigrama * 0.5f; // exige folga real, nao so empate
    printf("Perda ficou BEM abaixo do piso de unigrama (< 50%% dele)? %s\n\n",
           perdaAbaixoDoUnigrama ? "SIM (OK)" : "NAO (FALHOU)");

    // -----------------------------------------------------------------
    // 8. geracao: prompt = "alfa beta gama alfa beta gama", esperado
    //    continuar o padrao ciclico, NAO travar num token so
    // -----------------------------------------------------------------
    const char* prompt = "alfa beta gama alfa beta gama";
    Vetor<int> idsPrompt; idsPrompt.iniciar();
    tokParaGerar.codificar(prompt, (int)strlen(prompt), &idsPrompt);

    int gerados[12];
    modelo.gerarGuloso(idsPrompt.dados, idsPrompt.tam, gerados, 12);
    int tamTxt;
    char* textoGerado = tokParaGerar.decodificar(gerados, 12, &tamTxt);
    printf("[Geracao] prompt: \"%s\"\n", prompt);
    printf("[Geracao] continuacao: \"%s\"\n", textoGerado);

    // checagem de colapso: conta se mais de 80% dos ids gerados sao iguais
    int contagemMax = 0;
    std::map<int,int> freqGerados;
    for(int i = 0; i < 12; i++) freqGerados[gerados[i]]++;
    for(auto& kv : freqGerados) if(kv.second > contagemMax) contagemMax = kv.second;
    bool colapsou = contagemMax > 12 * 0.8f;
    printf("Geracao colapsou num unico token repetido? %s\n\n", colapsou ? "SIM (RUIM)" : "NAO (OK)");

    free(textoGerado);
    idsPrompt.liberar();

    printf("=== RESUMO ===\n");
    printf("Sanidade de dados (ver texto decodificado acima manualmente)\n");
    printf("Perda bem abaixo do piso de unigrama: %s\n", perdaAbaixoDoUnigrama ? "OK" : "FALHOU");
    printf("Geracao nao colapsou: %s\n", !colapsou ? "OK" : "FALHOU");

    bool tudoOk = perdaAbaixoDoUnigrama && !colapsou;
    printf("\nRESULTADO GERAL: %s\n", tudoOk ? "PASSOU" : "FALHOU");

    otim.liberar();
    free(treino.dados);
    return tudoOk ? 0 : 1;
}
