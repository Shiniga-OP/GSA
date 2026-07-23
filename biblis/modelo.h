// biblis/modelo.h
// Modelo completo: Embedding -> N x BlocoTransformer -> Densa(logits)
//
// prop():
//   ids[seq] -> Embedding -> x0[seq*dim]
//   x0 -> bloco[0] -> x1 -> bloco[1] -> ... -> bloco[N-1] -> xN[seq*dim]
//   xN -> Densa(dim,vocab) por token -> logits[seq*vocab]
//
// retroprop(): caminho inverso, cross-entropy -> Densa -> blocos (ordem reversa) -> Embedding
//
// perda: cross-entropy padrao, softmax aplicado aqui(nao dentro da Densa final)
#pragma once
#include "camadas/camada.h"
#include "camadas/embedding.h"
#include "camadas/transformer.h"
#include "camadas/densa.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>

struct Modelo {
    int vocab;
    int dim;
    int dimFF;
    int nCab;
    int nCamadas;
    int seqMax;
    int seqAtual;

    Embedding* emb;
    BlocoTransformer** blocos;
    Densa* saida;

    // buffers de ativacao intermediaria: uma por "fronteira" entre camadas (nCamadas+1 no total)
    // ativ[0] = saida do embedding, ativ[i] = saida do bloco i-1, ativ[nCamadas] = entrada da Densa final
    float** ativ; // [nCamadas+1] ponteiros, cada um [seqMax*dim]

    float* logits; // [seqMax * vocab]
    float* gradLogits; // [seqMax * vocab]

    // buffers de gradiente intermediario, mesma forma que ativ
    float** gradAtiv; // [nCamadas+1] ponteiros, cada um [seqMax*dim]

    // ponteiros pra Camada*, uteis pro otimizador (_totalParams/_coletarPtrs esperam Camada**)
    Camada** todasCamadas; // [1 (emb) + nCamadas (blocos) + 1 (saida)]
    int totalCamadas;

    Modelo(int vocab_, int dim_, int nCab_, int dimFF_, int nCamadas_, int seqMax_) {
        vocab = vocab_;
        dim = dim_;
        dimFF = dimFF_;
        nCab = nCab_;
        nCamadas = nCamadas_;
        seqMax = seqMax_;
        seqAtual = 1;

        emb = new Embedding(vocab, dim, seqMax);

        blocos = (BlocoTransformer**)malloc(nCamadas * sizeof(BlocoTransformer*));
        for(int i = 0; i < nCamadas; i++) {
            blocos[i] = new BlocoTransformer(dim, nCab, dimFF, seqMax);
        }
        saida = new Densa(dim, vocab, ""); // linear, softmax aplicado fora

        ativ = (float**)malloc((nCamadas + 1) * sizeof(float*));
        gradAtiv = (float**)malloc((nCamadas + 1) * sizeof(float*));
        for(int i = 0; i <= nCamadas; i++) {
            ativ[i] = (float*)malloc(seqMax * dim * sizeof(float));
            gradAtiv[i] = (float*)malloc(seqMax * dim * sizeof(float));
        }
        logits = (float*)malloc(seqMax * vocab * sizeof(float));
        gradLogits = (float*)malloc(seqMax * vocab * sizeof(float));

        totalCamadas = 1 + nCamadas + 1;
        todasCamadas = (Camada**)malloc(totalCamadas * sizeof(Camada*));
        todasCamadas[0] = emb;
        for(int i = 0; i < nCamadas; i++) todasCamadas[1 + i] = blocos[i];
        todasCamadas[totalCamadas - 1] = saida;
    }

    ~Modelo() {
        delete emb;
        for(int i = 0; i < nCamadas; i++) delete blocos[i];
        free(blocos);
        delete saida;
        for(int i = 0; i <= nCamadas; i++) { free(ativ[i]); free(gradAtiv[i]); }
        free(ativ);
        free(gradAtiv);
        free(logits);
        free(gradLogits);
        free(todasCamadas);
    }

    void inicializar(const char* metodo) {
        emb->inicializar(metodo);
        for(int i = 0; i < nCamadas; i++) blocos[i]->inicializar(metodo);
        saida->inicializar(metodo);
    }

    // define o comprimento de sequencia atual, propaga pra embedding e blocos
    void defSeq(int seq) {
        seqAtual = seq;
        emb->tamSeq = seq;
        for(int i = 0; i < nCamadas; i++) blocos[i]->defSeq(seq);
    }

    void zerarGrad() {
        for(int c = 0; c < totalCamadas; c++) todasCamadas[c]->zerarGrad();
    }

    // prop: ids[seqAtual](int*, reinterpretado como float* pra bater com Embedding)
    //       -> logits[seqAtual * vocab]
    // chamador deve ter chamado defSeq(seq) antes
    void prop(const int* ids) {
        int seq = seqAtual;

        emb->prop((const float*)ids, ativ[0]);

        for(int i = 0; i < nCamadas; i++) {
            blocos[i]->prop(ativ[i], ativ[i + 1]);
        }
        // Densa final: por token, dim -> vocab
        for(int t = 0; t < seq; t++) {
            saida->prop(ativ[nCamadas] + t * dim, logits + t * vocab);
        }
    }
    // perdaCrossEntropy: aplica softmax sobre logits[t] e calcula -log(p[alvo[t]])
    // alvos[seq]: id do token esperado em cada posicao
    // preenche gradLogits(dL/dlogits) para uso em retroprop()
    // retorna perda media sobre a sequencia
    float perdaCrossEntropy(const int* alvos) {
        int seq = seqAtual;
        float perdaTotal = 0.0f;

        for(int t = 0; t < seq; t++) {
            float* lg = logits + t * vocab;
            float* gl = gradLogits + t * vocab;

            // softmax estavel
            float mx = lg[0];
            for(int v = 1; v < vocab; v++) {
                if(lg[v] > mx) mx = lg[v];
            }
            float soma = 0.0f;
            for(int v = 0; v < vocab; v++) {
                gl[v] = expf(lg[v] - mx); // reaproveita gl como buffer temporario de exp
                soma += gl[v];
            }
            float invSoma = 1.0f / soma;

            int alvo = alvos[t];
            for(int v = 0; v < vocab; v++) {
                float p = gl[v] * invSoma; // probabilidade softmax
                gl[v] = p - (v == alvo ? 1.0f : 0.0f); // dL/dlogit = p - onehot
            }
            float pAlvo = expf(lg[alvo] - mx) * invSoma;
            // clamp evita log(0) por erro de arredondamento
            if(pAlvo < 1e-9f) pAlvo = 1e-9f;
            perdaTotal += -logf(pAlvo);
        }
        return perdaTotal / (float)seq;
    }
    // retroprop: caminho inverso completo, a partir de gradLogits (ja preenchido
    // por perdaCrossEntropy). Acumula gradientes em todas as camadas.
    void retroprop() {
        int seq = seqAtual;

        // grad atraves da Densa final, por token: gradLogits -> gradAtiv[nCamadas]
        for(int t = 0; t < seq; t++) {
            saida->retroprop(gradLogits + t * vocab, gradAtiv[nCamadas] + t * dim);
        }
        // grad atraves dos blocos, em ordem reversa
        for(int i = nCamadas - 1; i >= 0; i--) {
            blocos[i]->retroprop(gradAtiv[i + 1], gradAtiv[i]);
        }
        // grad atraves do embedding (gradEntrada ignorado: ids nao tem gradiente)
        emb->retroprop(gradAtiv[0], nullptr);
    }
    // gerar: sampling autoregressivo guloso a partir de um entrada de ids.
    // entrada[tamEntrada] -> escreve tamGerar novos ids em saidaIds (nao inclui o entrada)
    // usa apenas prop(), sem retroprop. respeita seqMax(janela deslizante simples).
    // =========================================================================
    void gerarGuloso(const int* entrada, int tamEntrada, int* saidaIds, int tamGerar) {
        int* buf = (int*)malloc(seqMax * sizeof(int));
        int tamBuf = tamEntrada < seqMax ? tamEntrada : seqMax;
        // copia so os ultimos seqMax ids da entrada, se maior que a janela
        memcpy(buf, entrada + (tamEntrada - tamBuf), tamBuf * sizeof(int));

        for(int g = 0; g < tamGerar; g++) {
            defSeq(tamBuf);
            prop(buf);

            float* ultimoLogit = logits + (tamBuf - 1) * vocab;
            int melhor = 0;
            float melhorVal = ultimoLogit[0];
            for(int v = 1; v < vocab; v++) {
                if(ultimoLogit[v] > melhorVal) {
                    melhorVal = ultimoLogit[v]; melhor = v;
                }
            }
            saidaIds[g] = melhor;

            if(tamBuf < seqMax) {
                buf[tamBuf] = melhor;
                tamBuf++;
            } else {
                // janela cheia: desliza (descarta o token mais antigo)
                memmove(buf, buf + 1, (seqMax - 1) * sizeof(int));
                buf[seqMax - 1] = melhor;
            }
        }
        free(buf);
    }
};