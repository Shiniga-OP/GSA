// biblis/camadas/norm.h
// normaliza cada vetor de dimensão independentemente:
// y = gamma * (x - media) / sqrt(variancia + eps) + beta
// parametros treinaveis: gamma[dim](escala), beta[dim](vies)
// gradientes: acumulados em gradGamma e gradBeta
#pragma once
#include "camada.h"
#include "../inicias.h"
#include <string.h>
#include <math.h>

struct Norm : Camada {
    int dim; // tamanho do vetor normalizado
    float eps; // epsilon de estabilidade numerica
    int seqMax; // pos maximas de estado guardadas por token
    int pos; // pos atual dentro do ciclo de chamadas de prop (0..seqMax-1)
    float* gamma; // escala treinável [dim], inicializado em 1
    float* beta; // viés treinável [dim], inicializado em 0
    float* gradGamma;
    float* gradBeta;

    // buffers do propagação para retropropagação, por token
    float* ultEnt; // cópia da entrada [seqMax*dim]
    float* ultNorm; // x normalizado (antes de gamma/beta) [seqMax*dim]
    float* ultMedia; // media do propagação [seqMax]
    float* ultVarInv; // 1/sqrt(var+eps) do propagação [seqMax]

    Norm(int dimensao, float epsilon = 1e-5f, int seqMaxNorm = 1) {
        dim = dimensao;
        eps = epsilon;
        seqMax = seqMaxNorm;
        pos = 0;
        gamma = (float*)malloc(dim * sizeof(float));
        beta = (float*)calloc(dim, sizeof(float));
        gradGamma = (float*)calloc(dim, sizeof(float));
        gradBeta = (float*)calloc(dim, sizeof(float));
        ultEnt = (float*)malloc(seqMax * dim * sizeof(float));
        ultNorm = (float*)malloc(seqMax * dim * sizeof(float));
        ultMedia = (float*)calloc(seqMax, sizeof(float));
        ultVarInv = (float*)malloc(seqMax * sizeof(float));
        for(int p = 0; p < seqMax; p++) ultVarInv[p] = 1.0f;
        // gamma = 1, beta = 0
        iniConstante(gamma, dim, 1.0f);
    }

    ~Norm() override {
        free(gamma);
        free(beta);
        free(gradGamma);
        free(gradBeta);
        free(ultEnt);
        free(ultNorm);
        free(ultMedia);
        free(ultVarInv);
    }
    // reinicia o contador de posição pra um novo ciclo de tokens (chamado no
    // inicio de cada prop()/retroprop() em loop de sequência)
    void defPos(int p) { pos = p; }

    // inicializar() reinicia gamma=1, beta=0(metodo ignorado)
    void inicializar(const char* /*metodo*/) override {
        iniConstante(gamma, dim, 1.0f);
        iniZeros(beta, dim);
    }
    // prop: normaliza entrada[dim] -> saida[dim]
    // grava ultEnt, ultNorm, ultMedia, ultVarInv para retropropagação
    void prop(const float* entrada, float* saida) override {
        int p = pos;
        float* eEnt = ultEnt + p*dim;
        float* eNorm = ultNorm + p*dim;
        memcpy(eEnt, entrada, dim * sizeof(float));

        // media com desenrolar x4
        float media = 0.0f;
        int i = 0;
        for(; i <= dim - 4; i += 4) {
            media += entrada[i] + entrada[i+1] + entrada[i+2] + entrada[i+3];
        }
        for(; i < dim; i++) media += entrada[i];
        media /= (float)dim;
        ultMedia[p] = media;

        // variancia com desenrolar x4
        float var = 0.0f;
        i = 0;
        for(; i <= dim - 4; i += 4) {
            float d0 = entrada[i]   - media, d1 = entrada[i+1] - media;
            float d2 = entrada[i+2] - media, d3 = entrada[i+3] - media;
            var += d0*d0 + d1*d1 + d2*d2 + d3*d3;
        }
        for(; i < dim; i++) {
            float d = entrada[i] - media; var += d*d;
        }
        var /= (float)dim;
        float varInv = 1.0f / sqrtf(var + eps);
        ultVarInv[p] = varInv;

        // normalizar + escala/vies com desenrolar x4
        i = 0;
        for(; i <= dim - 4; i += 4) {
            float n0 = (entrada[i] - media) * varInv;
            float n1 = (entrada[i+1] - media) * varInv;
            float n2 = (entrada[i+2] - media) * varInv;
            float n3 = (entrada[i+3] - media) * varInv;
            eNorm[i] = n0; saida[i] = gamma[i] * n0 + beta[i];
            eNorm[i+1] = n1;
            saida[i+1] = gamma[i+1] * n1 + beta[i+1];
            eNorm[i+2] = n2;
            saida[i+2] = gamma[i+2] * n2 + beta[i+2];
            eNorm[i+3] = n3;
            saida[i+3] = gamma[i+3] * n3 + beta[i+3];
        }
        for(; i < dim; i++) {
            float xNorm = (entrada[i] - media) * varInv;
            eNorm[i] = xNorm;
            saida[i] = gamma[i] * xNorm + beta[i];
        }
        pos = (p + 1) % seqMax;
    }
    // retroprop: gradSaida[dim] -> gradEntrada[dim], acumula gradGamma/gradBeta
    // derivada analitica completa de LN(via cadeia sobre media e variancia)
    void retroprop(const float* gradSaida, float* gradEntrada) override {
        int p = pos;
        float* eNorm = ultNorm + p*dim;
        float varInv = ultVarInv[p];

        // acumula gradGamma e gradBeta com desenrolar x4
        int i = 0;
        for(; i <= dim - 4; i += 4) {
            gradGamma[i] += gradSaida[i] * eNorm[i];
            gradGamma[i+1] += gradSaida[i+1] * eNorm[i+1];
            gradGamma[i+2] += gradSaida[i+2] * eNorm[i+2];
            gradGamma[i+3] += gradSaida[i+3] * eNorm[i+3];
            gradBeta[i] += gradSaida[i];
            gradBeta[i+1] += gradSaida[i+1];
            gradBeta[i+2] += gradSaida[i+2];
            gradBeta[i+3] += gradSaida[i+3];
        }
        for(; i < dim; i++) {
            gradGamma[i] += gradSaida[i] * eNorm[i];
            gradBeta[i] += gradSaida[i];
        }
        if(!gradEntrada) {
            pos = (p + 1) % seqMax;
            return;
        }
        // dl/dxNorm[i] = gradSaida[i] * gamma[i]
        // somaA = Σ g[i], somaB = Σ g[i]*xNorm[i]
        float somaA = 0.0f, somaB = 0.0f;
        i = 0;
        for(; i <= dim - 4; i += 4) {
            float g0 = gradSaida[i] * gamma[i];
            float g1 = gradSaida[i+1] * gamma[i+1];
            float g2 = gradSaida[i+2] * gamma[i+2];
            float g3 = gradSaida[i+3] * gamma[i+3];
            somaA += g0 + g1 + g2 + g3;
            somaB += g0*eNorm[i] + g1*eNorm[i+1] + g2*eNorm[i+2] + g3*eNorm[i+3];
        }
        for(; i < dim; i++) {
            float g = gradSaida[i] * gamma[i];
            somaA += g;
            somaB += g * eNorm[i];
        }
        // dL/dx[i] = varInv/dim * (dim*g[i] - somaA - xNorm[i]*somaB)
        float escala = varInv / (float)dim;
        float dimF = (float)dim;
        i = 0;
        for(; i <= dim - 4; i += 4) {
            gradEntrada[i] = escala * (dimF * gradSaida[i] * gamma[i] - somaA - eNorm[i] * somaB);
            gradEntrada[i+1] = escala * (dimF * gradSaida[i+1] * gamma[i+1] - somaA - eNorm[i+1] * somaB);
            gradEntrada[i+2] = escala * (dimF * gradSaida[i+2] * gamma[i+2] - somaA - eNorm[i+2] * somaB);
            gradEntrada[i+3] = escala * (dimF * gradSaida[i+3] * gamma[i+3] - somaA - eNorm[i+3] * somaB);
        }
        for(; i < dim; i++) {
            gradEntrada[i] = escala * (dimF * gradSaida[i] * gamma[i] - somaA - eNorm[i] * somaB);
        }
        pos = (p + 1) % seqMax;
    }
    int numParams() override {
        return 2 * dim;
    }

    void params(float** saida, int* tams) override {
        saida[0] = gamma;
        tams[0] = dim;
        saida[1] = beta;
        tams[1] = dim;
    }

    void gradParams(float** saida, int* tams) override {
        saida[0] = gradGamma;
        tams[0] = dim;
        saida[1] = gradBeta;
        tams[1] = dim;
    }

    void zerarGrad() override {
        memset(gradGamma, 0, dim * sizeof(float));
        memset(gradBeta, 0, dim * sizeof(float));
    }
};