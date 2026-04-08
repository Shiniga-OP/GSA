// teste_norm.cpp
// Testes intensivos da Norm (Layer Normalization).
// Compilar: clang++ -O2 -std=c++11 -o teste_norm teste_norm.cpp -lm
// Todos os testes imprimem PASSOU ou FALHOU com diagnóstico.
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include "biblis/inicias.h"
#include "biblis/camadas/norm.h"

// utilidades
static int falhas = 0;
static int total = 0;

static void checar(const char* nome, bool cond, const char* detalhe = "") {
    total++;
    if(cond) {
        printf("  [PASSOU] %s\n", nome);
    } else {
        printf("  [FALHOU] %s  %s\n", nome, detalhe);
        falhas++;
    }
}

static bool aproxEq(float a, float b, float tol = 1e-4f) {
    return fabsf(a - b) <= tol;
}

static bool vetoresAprox(const float* a, const float* b, int n, float tol = 1e-4f) {
    for(int i = 0; i < n; i++) {
        if(fabsf(a[i] - b[i]) > tol) return false;
    }
    return true;
}

// gradiente numerico de saida[idc] em relação a entrada[param]
static float gradNumerico(Norm& cm, const float* ent, int entParam, int saidaidc, float h = 1e-3f) {
    float* e = (float*)malloc(cm.dim * sizeof(float));
    float* s1 = (float*)malloc(cm.dim * sizeof(float));
    float* s2 = (float*)malloc(cm.dim * sizeof(float));
    memcpy(e, ent, cm.dim * sizeof(float));

    e[entParam] = ent[entParam] + h;
    cm.prop(e, s1);
    float vp = s1[saidaidc];

    e[entParam] = ent[entParam] - h;
    cm.prop(e, s2);
    float vm = s2[saidaidc];

    free(e);
    free(s1);
    free(s2);
    return (vp - vm) / (2.0f * h);
}

// 1. saida tem media ≈ 0 e desvio ≈ 1 antes de gamma/beta
void testeMediaDesvioPadrao() {
    printf("\n[1] Media e desvio padrão da saída normalizada\n");
    srand(42);
    const int DIM = 64;
    Norm cm(DIM);
    // gamma=1, beta=0 -> saída deve ter media≈0, std≈1

    float ent[DIM], sai[DIM];
    for(int i = 0; i < DIM; i++) ent[i] = (float)rand() / RAND_MAXf * 10.0f - 5.0f;
    cm.prop(ent, sai);

    float media = 0.0f;
    for(int i = 0; i < DIM; i++) media += sai[i];
    media /= DIM;

    float var = 0.0f;
    for(int i = 0; i < DIM; i++) var += (sai[i]-media)*(sai[i]-media);
    var /= DIM;

    char buf[64];
    snprintf(buf, sizeof(buf), "media=%.6f var=%.6f", media, var);
    checar("media ≈ 0", fabsf(media) < 1e-5f, buf);
    checar("variancia ≈ 1", fabsf(var - 1.0f) < 1e-4f, buf);
}

// 2. entrada constante -> saida deve ser zero(normalização de vetor constante)
// com entrada constante, var=0, varInv=1/sqrt(eps)≈316
// Erro de arredondamento em (x-media) é ~ulp(valor)*sqrt(dim)
// tolerancia realista: ulp(3.14)*sqrt(32)*varInv ≈ 2.4e-7 * 5.6 * 316 ≈ 4e-4
void testeEntradaConstante() {
    printf("\n[2] Entrada constante -> saída zero\n");
    const int DIM = 32;
    Norm cm(DIM);
    float ent[DIM], sai[DIM];
    for(int i = 0; i < DIM; i++) ent[i] = 3.14f;
    cm.prop(ent, sai);

    bool ok = true;
    for(int i = 0; i < DIM; i++) {
        if(fabsf(sai[i]) > 5e-4f) { ok = false; break; }
    }
    checar("constante -> saída ≈ zero (tol=5e-4, limitada por eps float)", ok);
}

// 3. gamma e beta aplicados corretamente
void testeGammaBeta() {
    printf("\n[3] gamma e beta aplicados\n");
    const int DIM = 8;
    Norm cm(DIM);
    // define gamma=2, beta=1
    for(int i = 0; i < DIM; i++) { cm.gamma[i] = 2.0f; cm.beta[i] = 1.0f; }

    float ent[DIM] = {1,2,3,4,5,6,7,8};
    float sai[DIM];
    cm.prop(ent, sai);

    // calcula referencia manualmente
    float media = 0.0f;
    for(int i = 0; i < DIM; i++) media += ent[i];
    media /= DIM;
    float var = 0.0f;
    for(int i = 0; i < DIM; i++) var += (ent[i]-media)*(ent[i]-media);
    var /= DIM;
    float std = sqrtf(var + 1e-5f);

    float ref[DIM];
    for(int i = 0; i < DIM; i++) ref[i] = 2.0f * ((ent[i]-media)/std) + 1.0f;

    checar("saída com gamma=2 beta=1", vetoresAprox(sai, ref, DIM));
}

// 4. gradiente analitico vs numerico em relação a entrada
void testeGradEntrada() {
    printf("\n[4] Gradiente analítico vs numérico (entrada)\n");
    srand(7);
    const int DIM = 16;
    Norm cm(DIM);
    // gamma aleatorio
    for(int i = 0; i < DIM; i++) cm.gamma[i] = 0.5f + (float)rand()/RAND_MAXf;

    float ent[DIM];
    for(int i = 0; i < DIM; i++) ent[i] = (float)rand()/RAND_MAXf * 4.0f - 2.0f;

    // gradSaida = vetor de uns
    float gradSai[DIM], gradEnt[DIM];
    for(int i = 0; i < DIM; i++) gradSai[i] = 1.0f;

    cm.prop(ent, gradSai); // so para popular buffers(sobrescreve gradSai abaixo)
    float sai[DIM];
    cm.prop(ent, sai);
    for(int i = 0; i < DIM; i++) gradSai[i] = 1.0f;
    cm.zerarGrad();
    cm.retroprop(gradSai, gradEnt);

    // gradiente numerico: d(soma saidas)/d(ent[p])
    bool ok = true;
    for(int p = 0; p < DIM; p++) {
        float gNum = 0.0f;
        for(int k = 0; k < DIM; k++) {
            gNum += gradNumerico(cm, ent, p, k);
        }
        if(fabsf(gradEnt[p] - gNum) > 1e-3f) {
            printf("    p=%d analitico=%.6f numerico=%.6f\n", p, gradEnt[p], gNum);
            ok = false;
        }
    }
    checar("grad entrada analitico ≈ numerico", ok);
}

// 5. gradiente de gamma e beta(acumulação)
void testeGradGammaBeta() {
    printf("\n[5] Gradiente de gamma e beta\n");
    srand(13);
    const int DIM = 12;
    Norm cm(DIM);
    for(int i = 0; i < DIM; i++) cm.gamma[i] = 1.0f;

    float ent[DIM], sai[DIM], gradSai[DIM], gradEnt[DIM];
    for(int i = 0; i < DIM; i++) { ent[i] = (float)rand()/RAND_MAXf * 6.0f - 3.0f; gradSai[i] = 1.0f; }
    cm.prop(ent, sai);
    cm.zerarGrad();
    cm.retroprop(gradSai, gradEnt);

    // gradGamma[i] deve ser ≈ ultNorm[i](com gradSai=1)
    // gradBeta[i] deve ser 1
    bool okBeta = true, okGamma = true;
    for(int i = 0; i < DIM; i++) {
        if(fabsf(cm.gradBeta[i] - 1.0f) > 1e-5f) okBeta = false;
        // compara com xNorm via referencia numerica
        float gGNum = 0.0f;
        // d(soma saidas)/d(gamma[i]) = xNorm[i] * gradSai[i] = xNorm[i]
        // xNorm[i] = (ent[i]-media)/std
        float media = 0.0f;
        for(int j = 0; j < DIM; j++) media += ent[j];
        media /= DIM;
        float var = 0.0f;
        for(int j = 0; j < DIM; j++) var += (ent[j]-media)*(ent[j]-media);
        var /= DIM;
        gGNum = (ent[i]-media)/sqrtf(var+1e-5f);
        if(fabsf(cm.gradGamma[i] - gGNum) > 1e-4f) {
            printf("    i=%d gradGamma=%.6f esperado=%.6f\n", i, cm.gradGamma[i], gGNum);
            okGamma = false;
        }
    }
    checar("gradBeta = 1 (gradSai=1)", okBeta);
    checar("gradGamma ≈ xNorm", okGamma);
}

// 6. zerarGrad funciona
void testeZerarGrad() {
    printf("\n[6] zerarGrad\n");
    const int DIM = 8;
    Norm cm(DIM);
    float ent[DIM], sai[DIM], gs[DIM], ge[DIM];
    for(int i = 0; i < DIM; i++) { ent[i] = (float)i; gs[i] = 1.0f; }
    cm.prop(ent, sai);
    cm.retroprop(gs, ge);
    cm.zerarGrad();
    bool ok = true;
    for(int i = 0; i < DIM; i++) {
        if(cm.gradGamma[i] != 0.0f || cm.gradBeta[i] != 0.0f) { ok = false; break; }
    }
    checar("gradientes zerados", ok);
}

// 7. acumulação de gradientes em multiplos retropropagação
void testeAcumulacaoGrad() {
    printf("\n[7] Acumulação de gradientes (2 retropropagação)\n");
    const int DIM = 8;
    Norm cm(DIM);
    float ent[DIM], sai[DIM], gs[DIM], ge[DIM];
    for(int i = 0; i < DIM; i++) { ent[i] = (float)i * 0.5f; gs[i] = 1.0f; }
    cm.prop(ent, sai);

    cm.zerarGrad();
    cm.retroprop(gs, ge);
    float gg1[DIM], gb1[DIM];
    memcpy(gg1, cm.gradGamma, DIM*sizeof(float));
    memcpy(gb1, cm.gradBeta, DIM*sizeof(float));

    // segundo retropropagação com mesma entrada(sem zerar)
    cm.prop(ent, sai);
    cm.retroprop(gs, ge);

    bool ok = true;
    for(int i = 0; i < DIM; i++) {
        if(!aproxEq(cm.gradGamma[i], 2.0f * gg1[i]) ||
           !aproxEq(cm.gradBeta[i],  2.0f * gb1[i])) {
            ok = false; break;
        }
    }
    checar("gradientes acumulados corretamente (2x)", ok);
}

// 8. epsilon previne divisão por zero(entrada quase constante)
void testeEpsilon() {
    printf("\n[8] Estabilidade numérica com variância ~0\n");
    const int DIM = 16;
    Norm cm(DIM, 1e-5f);
    float ent[DIM], sai[DIM];
    // quase constante
    for(int i = 0; i < DIM; i++) ent[i] = 1.0f + 1e-9f * (float)i;
    cm.prop(ent, sai);
    bool ok = true;
    for(int i = 0; i < DIM; i++) {
        if(!isfinite(sai[i])) { ok = false; break; }
    }
    checar("saída finita com variância ~0", ok);
}

// 9. gradiente finito com entrada quase constante
void testeGradEstabilidade() {
    printf("\n[9] Gradiente finito com variância ~0\n");
    const int DIM = 16;
    Norm cm(DIM, 1e-5f);
    float ent[DIM], sai[DIM], gs[DIM], ge[DIM];
    for(int i = 0; i < DIM; i++) { ent[i] = 1.0f; gs[i] = 1.0f; }
    cm.prop(ent, sai);
    cm.zerarGrad();
    cm.retroprop(gs, ge);
    bool ok = true;
    for(int i = 0; i < DIM; i++) {
        if(!isfinite(ge[i]) || !isfinite(cm.gradGamma[i])) { ok = false; break; }
    }
    checar("gradientes finitos com variância ~0", ok);
}

// 10. numParams e interface params/gradParams
void testeInterface() {
    printf("\n[10] Interface Camada (numParams, params, gradParams)\n");
    const int DIM = 24;
    Norm cm(DIM);
    checar("numParams = 2*dim", cm.numParams() == 2*DIM);

    float* p[2]; int t[2];
    cm.params(p, t);
    checar("params[0] = gamma", p[0] == cm.gamma);
    checar("params[1] = beta", p[1] == cm.beta);
    checar("tams[0] = dim", t[0] == DIM);
    checar("tams[1] = dim", t[1] == DIM);

    float* g[2]; int tg[2];
    cm.gradParams(g, tg);
    checar("gradParams[0] = gradGamma", g[0] == cm.gradGamma);
    checar("gradParams[1] = gradBeta",  g[1] == cm.gradBeta);
}

// 11. gradEntrada nulo não crasha
void testeGradEntradaNulo() {
    printf("\n[11] gradEntrada nullptr não crasha\n");
    const int DIM = 8;
    Norm cm(DIM);
    float ent[DIM], sai[DIM], gs[DIM];
    for(int i = 0; i < DIM; i++) { ent[i] = (float)i; gs[i] = 1.0f; }
    cm.prop(ent, sai);
    cm.zerarGrad();
    cm.retroprop(gs, nullptr);
    checar("retroprop com gradEntrada=nullptr", true);
}

// 12. dimensão grande: aleatoria, verifica media≈0 e std≈1
void testeDimensaoGrande() {
    printf("\n[12] Dimensão grande (dim=512)\n");
    srand(99);
    const int DIM = 512;
    Norm cm(DIM);
    float* ent = (float*)malloc(DIM*sizeof(float));
    float* sai = (float*)malloc(DIM*sizeof(float));
    for(int i = 0; i < DIM; i++) ent[i] = (float)rand()/RAND_MAXf * 20.0f - 10.0f;
    cm.prop(ent, sai);

    float media = 0.0f;
    for(int i = 0; i < DIM; i++) media += sai[i];
    media /= DIM;
    float var = 0.0f;
    for(int i = 0; i < DIM; i++) var += (sai[i]-media)*(sai[i]-media);
    var /= DIM;

    char buf[64];
    snprintf(buf, sizeof(buf), "media=%.8f var=%.8f", media, var);
    checar("media ≈ 0 (dim=512)", fabsf(media) < 1e-5f, buf);
    checar("variancia ≈ 1 (dim=512)", fabsf(var - 1.0f) < 1e-4f, buf);

    // retropropagação
    float* gs = (float*)malloc(DIM*sizeof(float));
    float* ge = (float*)malloc(DIM*sizeof(float));
    for(int i = 0; i < DIM; i++) gs[i] = 1.0f;
    cm.zerarGrad();
    cm.retroprop(gs, ge);

    bool okGe = true;
    for(int i = 0; i < DIM; i++) if(!isfinite(ge[i])) { okGe = false; break; }
    checar("gradientes finitos (dim=512)", okGe);

    free(ent);
    free(sai);
    free(gs);
    free(ge);
}

// 13. invariancia a translação da entrada
void testeInvarianciaTranslacao() {
    printf("\n[13] Invariância à translação\n");
    srand(55);
    const int DIM = 32;
    Norm cm(DIM);
    float ent[DIM], ent2[DIM], sai1[DIM], sai2[DIM];
    for(int i = 0; i < DIM; i++) ent[i] = (float)rand()/RAND_MAXf * 4.0f;
    for(int i = 0; i < DIM; i++) ent2[i] = ent[i] + 100.0f;

    cm.prop(ent, sai1);
    cm.prop(ent2, sai2);
    checar("saída invariante à translação", vetoresAprox(sai1, sai2, DIM));
}

// 14. invariancia a escala da entrada
void testeInvarianciaEscala() {
    printf("\n[14] Invariância à escala\n");
    srand(77);
    const int DIM = 32;
    Norm cm(DIM);
    float ent[DIM], ent2[DIM], sai1[DIM], sai2[DIM];
    for(int i = 0; i < DIM; i++) ent[i] = (float)rand()/RAND_MAXf * 4.0f - 2.0f;
    for(int i = 0; i < DIM; i++) ent2[i] = ent[i] * 5.0f;

    cm.prop(ent, sai1);
    cm.prop(ent2, sai2);
    checar("saída invariante à escala", vetoresAprox(sai1, sai2, DIM));
}

// 15. inicializar() reinicia gamma e beta
void testeInicializar() {
    printf("\n[15] inicializar() reinicia parâmetros\n");
    const int DIM = 16;
    Norm cm(DIM);
    for(int i = 0; i < DIM; i++) { cm.gamma[i] = 99.0f; cm.beta[i] = -99.0f; }
    cm.inicializar("qualquer");
    bool ok = true;
    for(int i = 0; i < DIM; i++) {
        if(cm.gamma[i] != 1.0f || cm.beta[i] != 0.0f) { ok = false; break; }
    }
    checar("gamma=1 e beta=0 após inicializar()", ok);
}

int main() {
    printf("=== Testes intensivos: Camada de Normalização ===\n");

    testeMediaDesvioPadrao();
    testeEntradaConstante();
    testeGammaBeta();
    testeGradEntrada();
    testeGradGammaBeta();
    testeZerarGrad();
    testeAcumulacaoGrad();
    testeEpsilon();
    testeGradEstabilidade();
    testeInterface();
    testeGradEntradaNulo();
    testeDimensaoGrande();
    testeInvarianciaTranslacao();
    testeInvarianciaEscala();
    testeInicializar();

    printf("\n=== Resultado: %d/%d passaram", total-falhas, total);
    if(falhas == 0) printf(" ✓ Tudo certo\n");
    else printf(" ✗ %d falha(s)\n", falhas);

    return falhas > 0 ? 1 : 0;
}