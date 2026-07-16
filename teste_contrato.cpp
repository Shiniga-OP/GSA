// teste_contrato.cpp
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "biblis/camadas/embedding.h"
#include "biblis/camadas/densa.h"
#include "biblis/camadas/norm.h"
#include "biblis/camadas/multicabeca.h"

static int passou = 0;
static int falhou = 0;

static void checar(const char* nome, bool ok) {
    if(ok) { printf("  [OK]    %s\n", nome); passou++; }
    else   { printf("  [FALHOU] %s\n", nome); falhou++; }
}

static void testeContratoEmbedding() {
    printf("\n[1] Contrato Embedding: prop copia tabela, retroprop acumula por id\n");
    const int VOCAB = 4, DIM = 4, SEQ_MAX = 8;
    Embedding emb(VOCAB, DIM, SEQ_MAX);

    for(int v = 0; v < VOCAB; v++)
        for(int d = 0; d < DIM; d++)
            emb.tabela[v * DIM + d] = (float)(v * 10 + d);

    memset(emb.gradTab, 0, VOCAB * DIM * sizeof(float));

    int ids[3] = {2, 0, 2};
    float saida[3 * DIM];
    emb.tamSeq = 3;
    emb.prop((const float*)ids, saida);

    bool copiaOk = true;
    for(int t = 0; t < 3; t++) {
        int id = ids[t];
        for(int d = 0; d < DIM; d++) {
            if(saida[t * DIM + d] != emb.tabela[id * DIM + d]) copiaOk = false;
        }
    }
    checar("prop() copia tabela[id] para saida[t]", copiaOk);

    float gradSai[3 * DIM];
    for(int i = 0; i < 3 * DIM; i++) gradSai[i] = 1.0f;
    emb.retroprop(gradSai, nullptr);

    bool acumOk = true;
    for(int d = 0; d < DIM; d++) {
        if(fabsf(emb.gradTab[2 * DIM + d] - 2.0f) > 1e-5f) acumOk = false;
    }
    checar("retroprop() acumula grad de duas posicoes do mesmo token", acumOk);
    checar("retroprop() nao contamina outros ids", fabsf(emb.gradTab[0 * DIM] - 1.0f) < 1e-5f);
    checar("retroprop() nao toca ids ausentes da sequencia", emb.gradTab[1 * DIM] == 0.0f);
}

static void testeContratoNorm() {
    printf("\n[2] Contrato Norm: saida normalizada, gradEntrada nao-nulo e mesma shape\n");
    const int DIM = 8;
    Norm nm(DIM);
    float entrada[DIM] = {1, 3, 5, 7, 9, 2, 4, 6};
    float saida[DIM];
    nm.prop(entrada, saida);

    float media = 0.0f;
    for(int i = 0; i < DIM; i++) media += saida[i];
    checar("prop() produz saida com media ~0 (gamma=1, beta=0)", fabsf(media/DIM) < 1e-5f);

    float var = 0.0f;
    for(int i = 0; i < DIM; i++) var += saida[i] * saida[i];
    checar("prop() produz saida com variancia ~1 (gamma=1, beta=0)", fabsf(var/DIM - 1.0f) < 1e-3f);

    float gradSai[DIM], gradEnt[DIM];
    for(int i = 0; i < DIM; i++) gradSai[i] = (float)i; 
    nm.retroprop(gradSai, gradEnt);

    float normaGrad = 0.0f;
    for(int i = 0; i < DIM; i++) normaGrad += gradEnt[i] * gradEnt[i];
    checar("retroprop() produz gradEntrada nao-nulo para gradSaida nao-nulo", normaGrad > 1e-10f);

    float somaGrad = 0.0f;
    for(int i = 0; i < DIM; i++) somaGrad += gradEnt[i];
    checar("retroprop() gradEntrada soma ~0 (propriedade analitica LN)", fabsf(somaGrad) < 1e-5f);
}

static void testeContratoNormAcumula() {
    printf("\n[3] Contrato Norm: gradGamma/gradBeta acumulam entre chamadas\n");
    const int DIM = 4;
    Norm nm(DIM);
    float ent[DIM] = {1, 2, 3, 4}, sai[DIM], gs[DIM] = {1, 1, 1, 1}, ge[DIM];
    nm.zerarGrad();
    nm.prop(ent, sai); nm.retroprop(gs, ge);
    nm.prop(ent, sai); nm.retroprop(gs, ge);
    checar("gradGamma acumula entre tokens (nao sobrescreve)", fabsf(nm.gradGamma[0]) > 1e-5f);
    checar("gradBeta acumula entre tokens (nao sobrescreve)", nm.gradBeta[0] == 2.0f);
}

static void testeContratoMCShape() {
    printf("\n[4] Contrato MultiCabeca: shape entrada/saida, gradEntrada nao-nulo\n");
    const int DIM = 8, NCAB = 2, SEQ_MAX = 8;
    MultiCabeca mc(DIM, NCAB, SEQ_MAX);
    int seqs[2] = {2, 4};
    for(int si = 0; si < 2; si++) {
        int seq = seqs[si];
        float ent[seq * DIM], sai[seq * DIM], gSai[seq * DIM], gEnt[seq * DIM];
        for(int i = 0; i < seq * DIM; i++) { ent[i] = 0.1f; gSai[i] = 1.0f; }
        mc.seqAtual = seq;
        mc.prop(ent, sai);
        mc.retroprop(gSai, gEnt);
        float nS = 0, nGE = 0;
        for(int i = 0; i < seq * DIM; i++) { nS += sai[i]*sai[i]; nGE += gEnt[i]*gEnt[i]; }
        char n1[64], n2[64];
        snprintf(n1, 64, "prop() saida nao-nula para seq=%d", seq);
        snprintf(n2, 64, "retroprop() gradEntrada nao-nulo para seq=%d", seq);
        checar(n1, nS > 1e-10f); checar(n2, nGE > 1e-10f);
    }
}

static void testeContratoResidualAcumula() {
    printf("\n[5] Contrato Residual: duas chamadas retroprop acumulam em gradTab\n");
    const int VOCAB = 2, DIM = 4, SEQ_MAX = 4;
    Embedding emb(VOCAB, DIM, SEQ_MAX);
    int ids[1] = {0}; float sai[DIM], gs[DIM] = {1, 1, 1, 1};
    emb.zerarGrad();
    emb.tamSeq = 1;
    emb.prop((const float*)ids, sai);
    emb.retroprop(gs, nullptr);
    emb.retroprop(gs, nullptr);
    checar("gradTab acumula dois caminhos de gradiente (nao sobrescreve)", emb.gradTab[0] == 2.0f);
}

static void testeContratoMCGradDistinto() {
    printf("\n[6] Contrato MultiCabeca: retroprop propaga gradiente diferenciado por token\n");
    const int DIM = 8, NCAB = 2, SEQ = 3, SEQ_MAX = 8;
    MultiCabeca mc(DIM, NCAB, SEQ_MAX);
    float ent[SEQ * DIM], sai[SEQ * DIM], gSai[SEQ * DIM], gEnt[SEQ * DIM];
    for(int i = 0; i < SEQ * DIM; i++) ent[i] = (float)rand() / RAND_MAX;
    memset(gSai, 0, sizeof(gSai));
    for(int d = 0; d < DIM; d++) gSai[0 * DIM + d] = 1.0f; // Gradiente só no token 0
    mc.seqAtual = SEQ;
    mc.prop(ent, sai);
    mc.retroprop(gSai, gEnt);
    bool diff = false;
    float n0 = 0, n1 = 0;
    for(int d = 0; d < DIM; d++) {
        n0 += gEnt[d]*gEnt[d]; n1 += gEnt[DIM+d]*gEnt[DIM+d];
        if(fabsf(gEnt[d] - gEnt[DIM+d]) > 1e-6f) diff = true;
    }
    checar("retroprop() diferencia gradiente por token (nao copia uniforme)", diff);
    checar("token alvo interage com outros (fluxo de gradiente detectado)", n1 > 1e-9f);
}

static void testeContratoDensaGradNumerico() {
    printf("\n[7] Contrato Densa: gradEntrada confere com diferenca finita\n");
    const int E = 4, S = 3; Densa d(E, S, "relu");
    float ent[E] = {0.5f, -0.3f, 0.8f, -0.1f}, sai[S], gS[S], gE[E];
    d.prop(ent, sai);
    for(int i = 0; i < S; i++) gS[i] = 2.0f * sai[i];
    d.zerarGrad(); d.retroprop(gS, gE);
    const float h = 1e-4f; bool ok = true;
    for(int i = 0; i < E; i++) {
        float eP[E], eM[E], sP[S], sM[S];
        memcpy(eP, ent, sizeof(ent)); memcpy(eM, ent, sizeof(ent));
        eP[i] += h; eM[i] -= h;
        d.prop(eP, sP); d.prop(eM, sM);
        float pP = 0, pM = 0;
        for(int j = 0; j < S; j++) { pP += sP[j]*sP[j]; pM += sM[j]*sM[j]; }
        if(fabsf(gE[i] - (pP-pM)/(2*h)) > 1e-3f) ok = false;
    }
    checar("gradEntrada analitico bate com diferenca finita (erro < 1e-3)", ok);
}

static void testeContratoNormGradNumerico() {
    printf("\n[8] Contrato Norm: gradEntrada confere com diferenca finita\n");
    const int DIM = 6; Norm nm(DIM);
    float ent[DIM] = {1.0f, 3.0f, -1.0f, 2.0f, 0.5f, -0.5f}, sai[DIM], gS[DIM], gE[DIM];
    for(int i = 0; i < DIM; i++) gS[i] = (i + 1.0f);
    nm.zerarGrad(); nm.prop(ent, sai); nm.retroprop(gS, gE);
    const float h = 1e-4f; bool ok = true;
    for(int i = 0; i < DIM; i++) {
        float eP[DIM], eM[DIM], sP[DIM], sM[DIM];
        memcpy(eP, ent, sizeof(ent)); memcpy(eM, ent, sizeof(ent));
        eP[i] += h; eM[i] -= h;
        nm.prop(eP, sP); nm.prop(eM, sM);
        float pP = 0, pM = 0;
        for(int j = 0; j < DIM; j++) { pP += sP[j]*gS[j]; pM += sM[j]*gS[j]; }
        if(fabsf(gE[i] - (pP-pM)/(2*h)) > 1e-2f) ok = false; // Tolerancia 1e-2 para FP32
    }
    checar("gradEntrada analitico bate com diferenca finita (erro < 1e-2)", ok);
}

int main() {
    printf("=== TESTES DE CONTRATO DE INTERFACE ENTRE CAMADAS ===\n");
    testeContratoEmbedding();
    testeContratoNorm();
    testeContratoNormAcumula();
    testeContratoMCShape();
    testeContratoResidualAcumula();
    testeContratoMCGradDistinto();
    testeContratoDensaGradNumerico();
    testeContratoNormGradNumerico();
    printf("\n===========================================\n");
    printf("Resultado: %d/19 passaram\n", passou);
    return falhou > 0;
}