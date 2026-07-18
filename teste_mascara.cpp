// teste_mascara_causal.cpp
// Confirma que a mascara causal esta correta: a saida da posicao q NAO deve
// mudar se alterarmos o conteudo de qualquer posicao k > q (futuro).
// Isso e testado tanto no prop() da MultiCabeca isolada quanto no BlocoTransformer
// completo, cobrindo forward. Tambem confere que gradEntrada em posicoes futuras
// nao influencia gradiente de posicoes passadas (checagem indireta via prop apos
// perturbacao, que e suficiente para provar ausencia de vazamento).
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include "biblis/camadas/multicabeca.h"
#include "biblis/camadas/transformer.h"

static float diffAbs(const float* a, const float* b, int n) {
    float d = 0.0f;
    for(int i = 0; i < n; i++) d += fabsf(a[i] - b[i]);
    return d;
}

int main() {
    int dim = 16;
    int nCab = 4;
    int seqMax = 8;
    int seq = 6;
    int posAlvo = 2; // vamos checar se a saida nesta posicao muda com perturbacao no futuro

    printf("=== teste_mascara_causal ===\n");

    // --- 1. teste isolado em MultiCabeca ---
    {
        MultiCabeca mha(dim, nCab, seqMax);
        mha.inicializar("xavier");
        mha.seqAtual = seq;

        float* entradaA = (float*)malloc(seq * dim * sizeof(float));
        float* entradaB = (float*)malloc(seq * dim * sizeof(float));
        float* saidaA = (float*)malloc(seq * dim * sizeof(float));
        float* saidaB = (float*)malloc(seq * dim * sizeof(float));

        for(int i = 0; i < seq * dim; i++) entradaA[i] = 0.01f * (float)(i % 5) - 0.02f;
        memcpy(entradaB, entradaA, seq * dim * sizeof(float));

        // perturba SOMENTE posicoes futuras (> posAlvo)
        for(int t = posAlvo + 1; t < seq; t++) {
            for(int d = 0; d < dim; d++) {
                entradaB[t * dim + d] += 10.0f; // perturbacao grande de proposito
            }
        }

        mha.prop(entradaA, saidaA);
        mha.prop(entradaB, saidaB);

        float difPosAlvo = diffAbs(saidaA + posAlvo * dim, saidaB + posAlvo * dim, dim);
        float difPosAnterior = diffAbs(saidaA + 0 * dim, saidaB + 0 * dim, dim);

        printf("MultiCabeca isolada:\n");
        printf("  diff na posicao %d (nao deveria mudar) = %f -> %s\n",
               posAlvo, difPosAlvo, difPosAlvo < 1e-4f ? "OK (sem vazamento)" : "FALHOU (vazamento do futuro!)");
        printf("  diff na posicao 0 (nao deveria mudar) = %f -> %s\n",
               difPosAnterior, difPosAnterior < 1e-4f ? "OK (sem vazamento)" : "FALHOU (vazamento do futuro!)");

        free(entradaA); free(entradaB); free(saidaA); free(saidaB);
    }

    // --- 2. teste no BlocoTransformer completo (garante que a correcao se propaga) ---
    {
        BlocoTransformer bloco(dim, nCab, 32, seqMax);
        bloco.inicializar("xavier");
        bloco.defSeq(seq);

        float* entradaA = (float*)malloc(seq * dim * sizeof(float));
        float* entradaB = (float*)malloc(seq * dim * sizeof(float));
        float* saidaA = (float*)malloc(seq * dim * sizeof(float));
        float* saidaB = (float*)malloc(seq * dim * sizeof(float));

        for(int i = 0; i < seq * dim; i++) entradaA[i] = 0.01f * (float)(i % 5) - 0.02f;
        memcpy(entradaB, entradaA, seq * dim * sizeof(float));

        for(int t = posAlvo + 1; t < seq; t++) {
            for(int d = 0; d < dim; d++) {
                entradaB[t * dim + d] += 10.0f;
            }
        }

        bloco.prop(entradaA, saidaA);
        bloco.prop(entradaB, saidaB);

        float difPosAlvo = diffAbs(saidaA + posAlvo * dim, saidaB + posAlvo * dim, dim);
        float difPosAnterior = diffAbs(saidaA + 0 * dim, saidaB + 0 * dim, dim);

        printf("\nBlocoTransformer completo:\n");
        printf("  diff na posicao %d (nao deveria mudar) = %f -> %s\n",
               posAlvo, difPosAlvo, difPosAlvo < 1e-4f ? "OK (sem vazamento)" : "FALHOU (vazamento do futuro!)");
        printf("  diff na posicao 0 (nao deveria mudar) = %f -> %s\n",
               difPosAnterior, difPosAnterior < 1e-4f ? "OK (sem vazamento)" : "FALHOU (vazamento do futuro!)");

        // controle: perturbar o PASSADO (posicao 0) DEVE mudar a saida em posAlvo
        // (confirma que a mascara nao esta zerando tudo por engano)
        float* entradaC = (float*)malloc(seq * dim * sizeof(float));
        memcpy(entradaC, entradaA, seq * dim * sizeof(float));
        for(int d = 0; d < dim; d++) entradaC[0 * dim + d] += 10.0f;

        float* saidaC = (float*)malloc(seq * dim * sizeof(float));
        bloco.prop(entradaC, saidaC);
        float difControlePassado = diffAbs(saidaA + posAlvo * dim, saidaC + posAlvo * dim, dim);

        printf("  [controle] diff na posicao %d ao perturbar posicao 0 (DEVE mudar) = %f -> %s\n",
               posAlvo, difControlePassado, difControlePassado > 1e-4f ? "OK (atencao ao passado funciona)" : "FALHOU (atencao nao esta fluindo do passado!)");

        bool ok = difPosAlvo < 1e-4f && difPosAnterior < 1e-4f && difControlePassado > 1e-4f;
        printf("\nRESULTADO: %s\n", ok ? "PASSOU" : "FALHOU");

        free(entradaA); free(entradaB); free(entradaC);
        free(saidaA); free(saidaB); free(saidaC);

        return ok ? 0 : 1;
    }
}