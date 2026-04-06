// biblis/camadas/densa.h
#pragma once
#include "camada.h"
#include "../inicias.h"
#include <string.h>
#include <math.h>

#define TILE 32

struct Densa : Camada {
    int nEnt; // numero de entradas
    int nSai; // numero de saidas
    float* pesos; // [nSai * nEnt], linha-maior: pesos[o*nEnt + i]
    float* bias; // [nSai]
    float* gradP; // gradiente de pesos acumulado
    float* gradB; // gradiente de bias acumulado
    float* ultEnt; // ultima entrada gravada em prop(), tamanho nEnt
    float* ultPre; // ultima pré-ativação gravada em prop(), tamanho nSai

    // === ciclo de vida ===
    Densa(int entradas, int saidas, const char* ativacao = "relu") {
        nEnt = entradas;
        nSai = saidas;
        pesos = (float*)malloc(nSai * nEnt * sizeof(float));
        bias = (float*)calloc(nSai, sizeof(float));
        gradP = (float*)calloc(nSai * nEnt, sizeof(float));
        gradB = (float*)calloc(nSai, sizeof(float));
        ultEnt = (float*)malloc(nEnt * sizeof(float));
        ultPre = (float*)malloc(nSai * sizeof(float));
        defAtivacao(ativacao);
        inicializar("xavier");
    }

    ~Densa() override {
        free(pesos);
        free(bias);
        free(gradP);
        free(gradB);
        free(ultEnt);
        free(ultPre);
    }

    // === inicialização ===
    void inicializar(const char* metodo) override {
        if (strcmp(metodo, "xavier") == 0) iniXavier(pesos, nSai * nEnt, nEnt, nSai);
        else if(strcmp(metodo, "he") == 0) iniHe(pesos, nSai * nEnt, nEnt);
        else if(strcmp(metodo, "zeros") == 0) iniZeros(pesos, nSai * nEnt);
        else if(strcmp(metodo, "constante") == 0) iniConstante(pesos, nSai * nEnt, 0.01f);
        else iniXavier(pesos, nSai * nEnt, nEnt, nSai);
        iniZeros(bias, nSai);
    }

    // === propagação direta ===
    // saida[o] = ativa(soma_i(pesos[o*nEnt+i] * entrada[i]) + bias[o])
    // GEMV com loop tile(blocos TILE*TILE) + desenrolamento x4
    void prop(const float* entrada, float* saida) override {
        memcpy(ultEnt, entrada, nEnt * sizeof(float));
        memcpy(ultPre, bias, nSai * sizeof(float));

        for(int oBase = 0; oBase < nSai; oBase += TILE) {
            int oFim = oBase + TILE < nSai ? oBase + TILE : nSai;
            for(int iBase = 0; iBase < nEnt; iBase += TILE) {
                int iFim = iBase + TILE < nEnt ? iBase + TILE : nEnt;
                for(int o = oBase; o < oFim; o++) {
                    float soma = 0.0f;
                    const float* l  = pesos + o * nEnt + iBase;
                    const float* en = entrada + iBase;
                    int tam = iFim - iBase;
                    int k = 0;
                    for(; k <= tam - 4; k += 4) {
                        soma += l[k]*en[k] + l[k+1]*en[k+1] +
                        l[k+2]*en[k+2] + l[k+3]*en[k+3];
                    }
                    for(; k < tam; k++) {
                        soma += l[k] * en[k];
                    }
                    ultPre[o] += soma;
                }
            }
        }
        if(ativa) {
            for(int o = 0; o < nSai; o++) {
                saida[o] = ativa(ultPre[o]);
            }
        } else {
            memcpy(saida, ultPre, nSai * sizeof(float));
        }
    }

    // === retropropagação ===
    // gradSaida : dL/d(saida), tamanho nSai
    // gradEntrada: dL/d(entrada), tamanho nEnt(pode ser nullptr)
    // acumula em gradP e gradB, chame zerarGrad() antes do lote
    // todas as derivadas em ativas.h operam sobre x(pré-ativação)
    void retroprop(const float* gradSaida, float* gradEntrada) override {
        float* delta = (float*)alloca(nSai * sizeof(float));
        if(derivada) {
            for(int o = 0; o < nSai; o++) {
                delta[o] = gradSaida[o] * derivada(ultPre[o]);
            }
        } else {
            memcpy(delta, gradSaida, nSai * sizeof(float));
        }
        for(int o = 0; o < nSai; o++) {
            gradB[o] += delta[o];
        }
        for(int oBase = 0; oBase < nSai; oBase += TILE) {
            int oFim = oBase + TILE < nSai ? oBase + TILE : nSai;
            for(int iBase = 0; iBase < nEnt; iBase += TILE) {
                int iFim = iBase + TILE < nEnt ? iBase + TILE : nEnt;
                for(int o = oBase; o < oFim; o++) {
                    float d = delta[o];
                    float* gL = gradP + o * nEnt + iBase;
                    const float* eL = ultEnt + iBase;
                    int tam = iFim - iBase;
                    int k = 0;
                    for(; k <= tam - 4; k += 4) {
                        gL[k] += d * eL[k];
                        gL[k+1] += d * eL[k+1];
                        gL[k+2] += d * eL[k+2];
                        gL[k+3] += d * eL[k+3];
                    }
                    for(; k < tam; k++) {
                        gL[k] += d * eL[k];
                    }
                }
            }
        }
        if(gradEntrada) {
            memset(gradEntrada, 0, nEnt * sizeof(float));
            for(int oBase = 0; oBase < nSai; oBase += TILE) {
                int oFim = oBase + TILE < nSai ? oBase + TILE : nSai;
                for(int iBase = 0; iBase < nEnt; iBase += TILE) {
                    int iFim = iBase + TILE < nEnt ? iBase + TILE : nEnt;
                    for(int o = oBase; o < oFim; o++) {
                        float d = delta[o];
                        const float* l  = pesos + o * nEnt + iBase;
                        float* ge = gradEntrada + iBase;
                        int tam = iFim - iBase;
                        int k = 0;
                        for(; k <= tam - 4; k += 4) {
                            ge[k] += d * l[k];
                            ge[k+1] += d * l[k+1];
                            ge[k+2] += d * l[k+2];
                            ge[k+3] += d * l[k+3];
                        }
                        for(; k < tam; k++) {
                            ge[k] += d * l[k];
                        }
                    }
                }
            }
        }
    }

    // === clip de gradiente(norma L2) ===
    void cliparGrad(float maxNorma) {
        float norma2 = 0.0f;
        for(int i = 0; i < nSai * nEnt; i++) norma2 += gradP[i] * gradP[i];
        for(int o = 0; o < nSai; o++) norma2 += gradB[o] * gradB[o];
        float norma = sqrtf(norma2);
        if(norma > maxNorma) {
            float escala = maxNorma / norma;
            for(int i = 0; i < nSai * nEnt; i++) gradP[i] *= escala;
            for(int o = 0; o < nSai; o++) gradB[o] *= escala;
        }
    }

    // === SGD inline(sem otimizador externo) ===
    void atualizarSGD(float lr, int tamBatch = 1) {
        float inv = lr / (float)tamBatch;
        for(int i = 0; i < nSai * nEnt; i++) pesos[i] -= inv * gradP[i];
        for(int o = 0; o < nSai; o++) bias[o] -= inv * gradB[o];
    }

    // === interface Camada ===
    int numParams() override { return nSai * nEnt + nSai; }

    void params(float** saida, int* tams) override {
        saida[0] = pesos;
        tams[0] = nSai * nEnt;
        saida[1] = bias;
        tams[1] = nSai;
    }

    void gradParams(float** saida, int* tams) override {
        saida[0] = gradP;
        tams[0] = nSai * nEnt;
        saida[1] = gradB;
        tams[1] = nSai;
    }

    void zerarGrad() override {
        memset(gradP, 0, nSai * nEnt * sizeof(float));
        memset(gradB, 0, nSai * sizeof(float));
    }
};
#undef TILE