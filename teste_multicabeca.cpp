// teste_multicabeca.cpp
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>
#include "biblis/camadas/multicabeca.h"
#include "biblis/util.h"

// utilitario
static int passou = 0;
static int falhou = 0;

static void verificar(const char* nome, bool ok, float val = 0.0f) {
    if(ok) {
        printf("  [OK]  %s  (%.6f)\n", nome, val);
        passou++;
    } else {
        printf("  [FALHOU] %s  (%.6f)\n", nome, val);
        falhou++;
    }
}

// gradiente numerico via diferenças finitas centrais
// f(x+h) - f(x-c) / (2h)
static void gradNumerico(
    MultiCabeca& mc, int seq,
    const float* ent,
    float* paramAlvo, int idcParam, // ponteiro para o parâmetro e índice
    float* gradNum, int nGrad, // saída: um escalar por chamada
    int idcSaida = -1 // se >= 0, dL/dparam via L2 em saida[idcSaida]
) {
    // calcula gradiente numérico do parâmetro paramAlvo[idcParam]
    // em relação à perda L2 total
    float* sai = (float*)malloc(seq * mc.dim * sizeof(float));
    float c = 1e-3f;

    float orig = paramAlvo[idcParam];

    paramAlvo[idcParam] = orig + c;
    mc.seqAtual = seq;
    mc.prop(ent, sai);
    float fp = perdaL2(sai, seq * mc.dim);

    paramAlvo[idcParam] = orig - c;
    mc.seqAtual = seq;
    mc.prop(ent, sai);
    float fm = perdaL2(sai, seq * mc.dim);

    paramAlvo[idcParam] = orig;
    *gradNum = (fp - fm) / (2.0f * c);
    free(sai);
}

//  teste 1: verificação de gradiente numerico vs analitico: pesos
static void testeGradPesos() {
    printf("\n[Teste 1] Gradiente analítico vs numérico — pesos\n");
    int dim = 8, nCab = 2, seq = 3, seqMax = 8;
    MultiCabeca mc(dim, nCab, seqMax);
    mc.seqAtual = seq;
    srand(42);
    vetorAleatorio(mc.Pq, dim*dim, 0.3f);
    vetorAleatorio(mc.Pk, dim*dim, 0.3f);
    vetorAleatorio(mc.Pv, dim*dim, 0.3f);
    vetorAleatorio(mc.Po, dim*dim, 0.3f);

    float* ent = (float*)malloc(seq * dim * sizeof(float));
    float* sai = (float*)malloc(seq * dim * sizeof(float));
    float* gEnt = (float*)malloc(seq * dim * sizeof(float));
    vetorAleatorio(ent, seq * dim, 0.5f);

    // propagação
    mc.prop(ent, sai);
    // grad da perda L2: dL/d(saida[i]) = saida[i]
    mc.zerarGrad();
    mc.retroprop(sai, gEnt); // gradSaida = saida (d(soma s^2/2)/ds = s)

    // verificar alguns indices de cada matriz
    struct {
        const char* nome; float* P; float* gP;
    } matrizes[] = {
        {"Pq", mc.Pq, mc.gPq},
        {"Pk", mc.Pk, mc.gPk},
        {"Pv", mc.Pv, mc.gPv},
        {"Po", mc.Po, mc.gPo},
    };
    int indices[] = {0, 1, dim+1, 2*dim+3, dim*dim-1};
    for(int m = 0; m < 4; m++) {
        for(int ii = 0; ii < 5; ii++) {
            int idc = indices[ii];
            float gNum;
            gradNumerico(mc, seq, ent, matrizes[m].P, idc, &gNum, 1);
            float gAna = matrizes[m].gP[idc];
            float absErr = fabsf(gNum - gAna);
            float relErr = absErr / (fabsf(gNum) + fabsf(gAna) + 1e-8f);
            // passa se erro absoluto < 1e-5 OU erro relativo < 2%
            bool ok = absErr < 1e-5f || relErr < 0.02f;
            char nome[64];
            snprintf(nome, sizeof(nome), "%s[%d] num=%.5f ana=%.5f", matrizes[m].nome, idc, gNum, gAna);
            verificar(nome, ok, relErr);
        }
    }
    free(ent);
    free(sai);
    free(gEnt);
}

// teste 2: gradiente numerico vs analitico: entrada
static void testeGradEntrada() {
    printf("\n[Teste 2] Gradiente analítico vs numérico — entrada\n");
    int dim = 8, nCab = 2, seq = 3, seqMax = 8;
    MultiCabeca mc(dim, nCab, seqMax);
    mc.seqAtual = seq;
    srand(7);
    vetorAleatorio(mc.Pq, dim*dim, 0.3f);
    vetorAleatorio(mc.Pk, dim*dim, 0.3f);
    vetorAleatorio(mc.Pv, dim*dim, 0.3f);
    vetorAleatorio(mc.Po, dim*dim, 0.3f);

    float* ent = (float*)malloc(seq * dim * sizeof(float));
    float* sai = (float*)malloc(seq * dim * sizeof(float));
    float* gEnt = (float*)malloc(seq * dim * sizeof(float));
    vetorAleatorio(ent, seq * dim, 0.5f);

    mc.prop(ent, sai);
    mc.zerarGrad();
    mc.retroprop(sai, gEnt);

    int n = seq * dim;
    float h = 1e-3f;
    int indices[] = {0, 1, dim, dim+3, n-1, n/2};
    for(int ii = 0; ii < 6; ii++) {
        int idc = indices[ii];
        float orig = ent[idc];
        float saiP[64], saiM[64]; // dim*seq <= 64 aqui

        ent[idc] = orig + h;
        mc.seqAtual = seq;
        mc.prop(ent, saiP);
        float fp = perdaL2(saiP, n);

        ent[idc] = orig - h;
        mc.seqAtual = seq;
        mc.prop(ent, saiM);
        float fm = perdaL2(saiM, n);

        ent[idc] = orig;
        float gNum = (fp - fm) / (2.0f * h);
        float gAna = gEnt[idc];
        float err = fabsf(gNum - gAna) / (fabsf(gNum) + fabsf(gAna) + 1e-8f);
        char nome[64];
        snprintf(nome, sizeof(nome), "gEnt[%d] num=%.5f ana=%.5f", idc, gNum, gAna);
        verificar(nome, err < 0.01f, err);
    }
    free(ent);
    free(sai);
    free(gEnt);
}

// teste 3: RoPE preserva norma
static void testeRoPENorma() {
    printf("\n[Teste 3] RoPE preserva norma do vetor\n");
    int dCab = 16;
    srand(99);
    for(int pos = 0; pos < 8; pos++) {
        float v[16];
        vetorAleatorio(v, dCab, 1.0f);
        float n1 = norma(v, dCab);
        MultiCabeca::rope(v, pos, dCab);
        float n2 = norma(v, dCab);
        float err = fabsf(n1 - n2) / (n1 + 1e-8f);
        char nome[32];
        snprintf(nome, sizeof(nome), "norma pos=%d %.5f->%.5f", pos, n1, n2);
        verificar(nome, err < 1e-5f, err);
    }
}

// teste 4: RoPE grad(ropeGrad é inversa de rope)
static void testeRoPEGrad() {
    printf("\n[Teste 4] ropeGrad é inversa de RoPE\n");
    int dCab = 8;
    srand(13);
    for(int pos = 0; pos < 5; pos++) {
        float v[8], orig[8], gSai[8], gEnt[8];
        vetorAleatorio(v, dCab, 1.0f);
        memcpy(orig, v, dCab * sizeof(float));
        MultiCabeca::rope(v, pos, dCab);
        // gSai = v rotacionado, queremos recuperar orig
        MultiCabeca::ropeGrad(v, gEnt, pos, dCab);
        // gEnt deve ser igual a orig (pois a entrada pre-rope era orig e gSai = rope(orig))
        float err = normaDif(gEnt, orig, dCab) / (norma(orig, dCab) + 1e-8f);
        char nome[32];
        snprintf(nome, sizeof(nome), "rope^-1 pos=%d", pos);
        verificar(nome, err < 1e-5f, err);
    }
}

//  Teste 5: softmax invariante a shift de constante
static void testeSoftmaxEstavel() {
    printf("\n[Teste 5] Softmax estável — saída idêntica com shift\n");
    int dim = 4, nCab = 2, seq = 4, seqMax = 8;
    MultiCabeca mc1(dim, nCab, seqMax);
    MultiCabeca mc2(dim, nCab, seqMax);
    mc1.seqAtual = seq;
    mc2.seqAtual = seq;

    srand(55);
    vetorAleatorio(mc1.Pq, dim*dim, 0.3f);
    memcpy(mc2.Pq, mc1.Pq, dim*dim*sizeof(float));
    vetorAleatorio(mc1.Pk, dim*dim, 0.3f);
    memcpy(mc2.Pk, mc1.Pk, dim*dim*sizeof(float));
    vetorAleatorio(mc1.Pv, dim*dim, 0.3f);
    memcpy(mc2.Pv, mc1.Pv, dim*dim*sizeof(float));
    vetorAleatorio(mc1.Po, dim*dim, 0.3f);
    memcpy(mc2.Po, mc1.Po, dim*dim*sizeof(float));

    float ent1[16], ent2[16];
    vetorAleatorio(ent1, seq * dim, 1.0f);
    // ent2 = ent1 * 1000(vai produzir logits enormes, testa estabilidade)
    for(int i = 0; i < seq*dim; i++) ent2[i] = ent1[i] * 1000.0f;

    // para que a saida seja a mesma(proporcionalmente), usamos mesma entrada
    // aqui testamos que valores extremos não produzem NaN/Inf
    float sai1[16], sai2[16];
    mc1.prop(ent1, sai1);
    mc2.prop(ent2, sai2);

    bool semNaN1 = true, semNaN2 = true;
    for(int i = 0; i < seq*dim; i++) {
        if(!isfinite(sai1[i])) semNaN1 = false;
        if(!isfinite(sai2[i])) semNaN2 = false;
    }
    verificar("saida normal sem NaN/Inf", semNaN1, 0.0f);
    verificar("saida extrema sem NaN/Inf", semNaN2, 0.0f);
}

// teste 6: determinismo: prop() é deterministica
static void testeDeterminismo() {
    printf("\n[Teste 6] prop() determinística\n");
    int dim = 8, nCab = 2, seq = 4, seqMax = 8;
    MultiCabeca mc(dim, nCab, seqMax);
    mc.seqAtual = seq;
    srand(11);
    vetorAleatorio(mc.Pq, dim*dim, 0.3f);
    vetorAleatorio(mc.Pk, dim*dim, 0.3f);
    vetorAleatorio(mc.Pv, dim*dim, 0.3f);
    vetorAleatorio(mc.Po, dim*dim, 0.3f);

    float ent[32], sai1[32], sai2[32];
    vetorAleatorio(ent, seq*dim, 0.5f);

    mc.prop(ent, sai1);
    mc.prop(ent, sai2);

    float err = normaDif(sai1, sai2, seq*dim);
    verificar("duas props idênticas", err < 1e-10f, err);
}

// teste 7: convergencia: SGD minimiza L2 em poucas iterações
static void testeConvergencia() {
    printf("\n[Teste 7] Convergência SGD — perda L2 deve cair\n");
    int dim = 8, nCab = 2, seq = 3, seqMax = 8;
    MultiCabeca mc(dim, nCab, seqMax);
    mc.seqAtual = seq;
    srand(77);
    vetorAleatorio(mc.Pq, dim*dim, 0.4f);
    vetorAleatorio(mc.Pk, dim*dim, 0.4f);
    vetorAleatorio(mc.Pv, dim*dim, 0.4f);
    vetorAleatorio(mc.Po, dim*dim, 0.4f);

    float ent[24], sai[24];
    vetorAleatorio(ent, seq*dim, 1.5f);

    float taxa = 0.01f;
    float perdaInicial = 0.0f, perdaFinal = 0.0f;

    mc.prop(ent, sai);
    perdaInicial = perdaL2(sai, seq*dim);

    for(int iter = 0; iter < 200; iter++) {
        mc.zerarGrad();
        mc.prop(ent, sai);

        // objetivo: saida -> 0, então gradSaida = saida
        mc.retroprop(sai, nullptr);

        // SGD nos 4 pesos
        float* Ps[]  = {mc.Pq, mc.Pk, mc.Pv, mc.Po};
        float* gPs[] = {mc.gPq, mc.gPk, mc.gPv, mc.gPo};
        for(int m = 0; m < 4; m++) {
            for(int i = 0; i < dim*dim; i++) {
                Ps[m][i] -= taxa * gPs[m][i];
            }
        }
    }
    mc.prop(ent, sai);
    perdaFinal = perdaL2(sai, seq*dim);

    printf("    perda inicial=%.6f  final=%.6f\n", perdaInicial, perdaFinal);
    verificar("perda final < perda inicial", perdaFinal < perdaInicial * 0.5f,
    perdaFinal / perdaInicial);
}

//  Teste 8: zerarGrad() realmente zera
static void testeZerarGrad() {
    printf("\n[Teste 8] zerarGrad() zera todos os gradientes\n");
    int dim = 8, nCab = 2, seq = 3, seqMax = 8;
    MultiCabeca mc(dim, nCab, seqMax);
    mc.seqAtual = seq;
    srand(33);

    float ent[24], sai[24], gEnt[24];
    vetorAleatorio(ent, seq*dim, 0.5f);
    mc.prop(ent, sai);
    mc.retroprop(sai, gEnt);

    mc.zerarGrad();

    float n = norma(mc.gPq, dim*dim) + norma(mc.gPk, dim*dim)
    + norma(mc.gPv, dim*dim) + norma(mc.gPo, dim*dim);
    verificar("norma dos grads == 0 após zerarGrad()", n < 1e-10f, n);
}

// teste 9: sequencias de comprimento diferente não interferem
static void testeSequenciasIndependentes() {
    printf("\n[Teste 9] Sequências de comprimentos diferentes são independentes\n");
    int dim = 8, nCab = 2, seqMax = 8;
    MultiCabeca mc(dim, nCab, seqMax);
    srand(21);
    vetorAleatorio(mc.Pq, dim*dim, 0.3f);
    vetorAleatorio(mc.Pk, dim*dim, 0.3f);
    vetorAleatorio(mc.Pv, dim*dim, 0.3f);
    vetorAleatorio(mc.Po, dim*dim, 0.3f);

    float ent3[24], ent5[40];
    float sai3a[24], sai3b[24];
    float sai5[40];
    vetorAleatorio(ent3, 3*dim, 0.5f);
    vetorAleatorio(ent5, 5*dim, 0.5f);

    mc.seqAtual = 3;
    mc.prop(ent3, sai3a);

    // prop com seq=5 não deve contaminar resultado de seq=3
    mc.seqAtual = 5;
    mc.prop(ent5, sai5);

    mc.seqAtual = 3;
    mc.prop(ent3, sai3b);

    float err = normaDif(sai3a, sai3b, 3*dim);
    verificar("seq=3 idêntica antes e depois de seq=5", err < 1e-10f, err);
}

// teste 10: params() e gradParams() apontam para os buffers corretos
static void testeInterface() {
    printf("\n[Teste 10] params() e gradParams() apontam corretamente\n");
    int dim = 8, nCab = 2, seqMax = 8;
    MultiCabeca mc(dim, nCab, seqMax);

    float* p[4];
    int tp[4];
    float* g[4];
    int tg[4];
    mc.params(p, tp);
    mc.gradParams(g, tg);

    verificar("params[0] == Pq", p[0] == mc.Pq,  0.0f);
    verificar("params[1] == Pk", p[1] == mc.Pk,  0.0f);
    verificar("params[2] == Pv", p[2] == mc.Pv,  0.0f);
    verificar("params[3] == Po", p[3] == mc.Po,  0.0f);
    verificar("grads[0] == gPq", g[0] == mc.gPq, 0.0f);
    verificar("grads[1] == gPk", g[1] == mc.gPk, 0.0f);
    verificar("grads[2] == gPv", g[2] == mc.gPv, 0.0f);
    verificar("grads[3] == gPo", g[3] == mc.gPo, 0.0f);
    verificar("numParams() == 4*dim*dim", mc.numParams() == 4*dim*dim, (float)mc.numParams());
    for(int i = 0; i < 4; i++) {
        verificar("tams pesos", tp[i] == dim*dim, (float)tp[i]);
        verificar("tams grads", tg[i] == dim*dim, (float)tg[i]);
    }
}

int main() {
    srand((unsigned)time(nullptr));
    printf("=== Testes MultiCabeca ===\n");

    testeGradPesos();
    testeGradEntrada();
    testeRoPENorma();
    testeRoPEGrad();
    testeSoftmaxEstavel();
    testeDeterminismo();
    testeConvergencia();
    testeZerarGrad();
    testeSequenciasIndependentes();
    testeInterface();

    printf("\n=== Resultado: %d passou, %d falhou ===\n", passou, falhou);
    return falhou > 0 ? 1 : 0;
}