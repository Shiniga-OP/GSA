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

int main() {
    srand(42);

    // === 1. treino BPE ===
    int tamTexto;
    char* texto = FabricaDados::lerArquivoTexto("corpus_teste.txt", &tamTexto);
    if(!texto) {
        printf("FALHOU: nao leu corpus\n");
        return 1;
    }
    printf("corpus lido: %d bytes\n", tamTexto);

    TreinadorBPE treinador;
    treinador.treinar(texto, tamTexto, 120);
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

    // === 2. modelo pequeno, mas nao minusculo demais pra dar pra ver linguagem ===
    int dim = 64;
    int nCab = 4;
    int dimFF = 128;
    int nCamadas = 3;
    int seqMax = 32;

    Modelo modelo(vocab, dim, nCab, dimFF, nCamadas, seqMax);
    modelo.inicializar("xavier");
    printf("modelo criado: vocab=%d dim=%d nCab=%d dimFF=%d nCamadas=%d seqMax=%d\n",
        vocab, dim, nCab, dimFF, nCamadas, seqMax);

    AdamW otim;
    otim.iniciar(modelo.todasCamadas, modelo.totalCamadas, 3e-3f);

    // === 3. sequencias de treino: janelas deslizantes de seqMax+1 sobre os tokens ===
    int seq = seqMax;
    int stride = 4;
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
    // taxa fixa apos o warmup diverge (visto no log: perda cai, oscila, explode
    // no final). o AgendadorCosseno ja existia em otimizador.h mas nunca era
    // usado em lugar nenhum; aqui ele conduz a taxa do inicio ao fim do treino.
    int passos = 800;
    int aquecimento = 50;
    bool viuNan = false;

    AgendadorCosseno agenda;
    agenda.taxaMax = 3e-3f;
    agenda.taxaMin = 1e-5f;
    agenda.passosTotal = passos;
    agenda.aquecimento = aquecimento;

    modelo.defSeq(seq);

    for(int p = 0; p < passos; p++) {
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

        // gera uma amostra de texto a cada 100 passos, pra ver a linguagem evoluir
        if(p % 100 == 0 || p == passos - 1) {
            gerarTexto(modelo, tok, semente, tamSemente, 40, 0.8f, bufTexto, sizeof(bufTexto));
            printf("\n--- passo %d (perda=%.4f) ---\n%s\n", p, perda, bufTexto);
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
