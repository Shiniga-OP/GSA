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
    float* gamma; // escala treinável [dim], inicializado em 1
    float* beta; // viés treinável [dim], inicializado em 0
    float* gradGamma;
    float* gradBeta;

    // buffers do propagação para retropropagação
    float* ultEnt; // cópia da entrada [dim]
    float* ultNorm; // x normalizado (antes de gamma/beta) [dim]
    float ultMedia; // media do propagação
    float ultVarInv; // 1/sqrt(var+eps) do propagação

    Norm(int dimensao, float epsilon = 1e-5f) {
        dim = dimensao;
        eps = epsilon;
        gamma = (float*)malloc(dim * sizeof(float));
        beta = (float*)calloc(dim, sizeof(float));
        gradGamma = (float*)calloc(dim, sizeof(float));
        gradBeta = (float*)calloc(dim, sizeof(float));
        ultEnt = (float*)malloc(dim * sizeof(float));
        ultNorm = (float*)malloc(dim * sizeof(float));
        ultMedia = 0.0f;
        ultVarInv = 1.0f;
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
    }

    // inicializar() reinicia gamma=1, beta=0(metodo ignorado)
    void inicializar(const char* /*metodo*/) override {
        iniConstante(gamma, dim, 1.0f);
        iniZeros(beta, dim);
    }
    // prop: normaliza entrada[dim] -> saida[dim]
    // grava ultEnt, ultNorm, ultMedia, ultVarInv para retropropagação
    void prop(const float* entrada, float* saida) override {
        memcpy(ultEnt, entrada, dim * sizeof(float));

        // media com desenrolar x4
        float media = 0.0f;
        int i = 0;
        for(; i <= dim - 4; i += 4) {
            media += entrada[i] + entrada[i+1] + entrada[i+2] + entrada[i+3];
        }
        for(; i < dim; i++) media += entrada[i];
        media /= (float)dim;
        ultMedia = media;

        // variancia com desenrolar x4
        float var = 0.0f;
        i = 0;
        for(; i <= dim - 4; i += 4) {
            float d0 = entrada[i]   - media, d1 = entrada[i+1] - media;
            float d2 = entrada[i+2] - media, d3 = entrada[i+3] - media;
            var += d0*d0 + d1*d1 + d2*d2 + d3*d3;
        }
        for(; i < dim; i++) { float d = entrada[i] - media; var += d*d; }
        var /= (float)dim;
        float varInv = 1.0f / sqrtf(var + eps);
        ultVarInv = varInv;

        // normalizar + escala/vies com desenrolar x4
        i = 0;
        for(; i <= dim - 4; i += 4) {
            float n0 = (entrada[i] - media) * varInv;
            float n1 = (entrada[i+1] - media) * varInv;
            float n2 = (entrada[i+2] - media) * varInv;
            float n3 = (entrada[i+3] - media) * varInv;
            ultNorm[i] = n0; saida[i] = gamma[i] * n0 + beta[i];
            ultNorm[i+1] = n1;
            saida[i+1] = gamma[i+1] * n1 + beta[i+1];
            ultNorm[i+2] = n2;
            saida[i+2] = gamma[i+2] * n2 + beta[i+2];
            ultNorm[i+3] = n3;
            saida[i+3] = gamma[i+3] * n3 + beta[i+3];
        }
        for(; i < dim; i++) {
            float xNorm = (entrada[i] - media) * varInv;
            ultNorm[i] = xNorm;
            saida[i] = gamma[i] * xNorm + beta[i];
        }
    }

    // retroprop: gradSaida[dim] → gradEntrada[dim], acumula gradGamma/gradBeta
    // Derivada analítica completa de LN (via cadeia sobre média e variância).
    void retroprop(const float* gradSaida, float* gradEntrada) override {
        // acumula gradGamma e gradBeta com desenrolar x4
        int i = 0;
        for(; i <= dim - 4; i += 4) {
            gradGamma[i] += gradSaida[i] * ultNorm[i];
            gradGamma[i+1] += gradSaida[i+1] * ultNorm[i+1];
            gradGamma[i+2] += gradSaida[i+2] * ultNorm[i+2];
            gradGamma[i+3] += gradSaida[i+3] * ultNorm[i+3];
            gradBeta[i] += gradSaida[i];
            gradBeta[i+1] += gradSaida[i+1];
            gradBeta[i+2] += gradSaida[i+2];
            gradBeta[i+3] += gradSaida[i+3];
        }
        for(; i < dim; i++) {
            gradGamma[i] += gradSaida[i] * ultNorm[i];
            gradBeta[i] += gradSaida[i];
        }
        if(!gradEntrada) return;

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
            somaB += g0*ultNorm[i] + g1*ultNorm[i+1] + g2*ultNorm[i+2] + g3*ultNorm[i+3];
        }
        for(; i < dim; i++) {
            float g = gradSaida[i] * gamma[i];
            somaA += g;
            somaB += g * ultNorm[i];
        }
        // dL/dx[i] = varInv/dim * (dim*g[i] - somaA - xNorm[i]*somaB)
        float escala = ultVarInv / (float)dim;
        float dimF = (float)dim;
        i = 0;
        for(; i <= dim - 4; i += 4) {
            gradEntrada[i] = escala * (dimF * gradSaida[i] * gamma[i] - somaA - ultNorm[i] * somaB);
            gradEntrada[i+1] = escala * (dimF * gradSaida[i+1] * gamma[i+1] - somaA - ultNorm[i+1] * somaB);
            gradEntrada[i+2] = escala * (dimF * gradSaida[i+2] * gamma[i+2] - somaA - ultNorm[i+2] * somaB);
            gradEntrada[i+3] = escala * (dimF * gradSaida[i+3] * gamma[i+3] - somaA - ultNorm[i+3] * somaB);
        }
        for(; i < dim; i++) {
            gradEntrada[i] = escala * (dimF * gradSaida[i] * gamma[i] - somaA - ultNorm[i] * somaB);
        }
    }
    int numParams() override { return 2 * dim; }

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