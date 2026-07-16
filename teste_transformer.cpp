// teste_transformer.cpp
// valida BlocoTransformer: prop sem NaN/Inf, retroprop com gradientes nao-nulos,
// numParams consistente com as sub-camadas.
#include <cstdio>
#include <cmath>
#include "biblis/camadas/transformer.h"

static bool temNaoFinito(const float* v, int n) {
    for(int i = 0; i < n; i++) {
        if(!std::isfinite(v[i])) return true;
    }
    return false;
}

static float somaAbs(const float* v, int n) {
    float s = 0.0f;
    for(int i = 0; i < n; i++) s += fabsf(v[i]);
    return s;
}

int main() {
    int dim = 16;
    int nCab = 4;
    int dimFF = 64;
    int seqMax = 8;
    int seq = 5;

    printf("=== teste_transformer ===\n");
    printf("dim=%d nCab=%d dimFF=%d seqMax=%d seq=%d\n", dim, nCab, dimFF, seqMax, seq);

    BlocoTransformer bloco(dim, nCab, dimFF, seqMax);
    bloco.inicializar("xavier");
    bloco.defSeq(seq);

    // --- 1. checagem de numParams ---
    int esperado = 4*dim*dim          // MHA: Pq,Pk,Pv,Po
                 + 2*dim              // ln1: gamma,beta
                 + 2*dim              // ln2: gamma,beta
                 + (dim*dimFF + dimFF)   // ff1: pesos+bias
                 + (dimFF*dim + dim);    // ff2: pesos+bias
    int obtido = bloco.numParams();
    printf("numParams: obtido=%d esperado=%d -> %s\n",
           obtido, esperado, obtido == esperado ? "OK" : "FALHOU");

    // --- 2. prop com entrada sintetica ---
    float* entrada = (float*)malloc(seq * dim * sizeof(float));
    float* saida   = (float*)malloc(seq * dim * sizeof(float));
    for(int i = 0; i < seq * dim; i++) {
        entrada[i] = 0.01f * (float)(i % 7) - 0.03f;
    }

    bloco.prop(entrada, saida);

    bool saidaRuim = temNaoFinito(saida, seq * dim);
    printf("prop: saida contem NaN/Inf? %s\n", saidaRuim ? "SIM (FALHOU)" : "nao (OK)");
    printf("prop: soma|saida| = %f (esperado != 0)\n", somaAbs(saida, seq * dim));

    // --- 3. retroprop com gradSaida sintetico (ex: erro = saida - alvo) ---
    float* gradSaida   = (float*)malloc(seq * dim * sizeof(float));
    float* gradEntrada = (float*)malloc(seq * dim * sizeof(float));
    for(int i = 0; i < seq * dim; i++) gradSaida[i] = 0.1f;

    bloco.zerarGrad();
    bloco.retroprop(gradSaida, gradEntrada);

    bool gradRuim = temNaoFinito(gradEntrada, seq * dim);
    float somaGradEnt = somaAbs(gradEntrada, seq * dim);
    printf("retroprop: gradEntrada contem NaN/Inf? %s\n", gradRuim ? "SIM (FALHOU)" : "nao (OK)");
    printf("retroprop: soma|gradEntrada| = %f (esperado != 0)\n", somaGradEnt);

    // --- 4. checagem de que os gradientes de parametros nao ficaram todos zero ---
    float* ptrsP[16];
    int tamsP[16];
    bloco.gradParams(ptrsP, tamsP);
    float somaGradParams = 0.0f;
    for(int g = 0; g < bloco.grupos; g++) {
        somaGradParams += somaAbs(ptrsP[g], tamsP[g]);
    }
    printf("retroprop: soma|gradParams| (todos os %d grupos) = %f (esperado != 0)\n",
           bloco.grupos, somaGradParams);

    // --- veredito final ---
    bool ok = (obtido == esperado) && !saidaRuim && !gradRuim
            && somaAbs(saida, seq*dim) > 0.0f
            && somaGradEnt > 0.0f
            && somaGradParams > 0.0f;

    printf("\nRESULTADO: %s\n", ok ? "PASSOU" : "FALHOU");

    free(entrada); free(saida);
    free(gradSaida); free(gradEntrada);
    return ok ? 0 : 1;
}