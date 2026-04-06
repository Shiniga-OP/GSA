// teste_densa.cpp
// Compila: g++ -O2 -o teste_densa teste_densa.cpp -lm
// ou:      clang++ -O2 -o teste_densa teste_densa.cpp -lm

#include <stdio.h>
#include <math.h>
#include <string.h>
#include <stdlib.h>
#include <float.h>
#include "biblis/camadas/densa.h"

// === utilitarios de teste ===

static int _total = 0, _passou = 0;

static void check(const char* nome, int cond) {
    _total++;
    if(cond) { _passou++; printf("  [OK]  %s\n", nome); }
    else printf("  [FALHOU] %s\n", nome);
}

static float absF(float x) { return x < 0 ? -x : x; }

static bool proximos(float a, float b, float tol = 1e-4f) {
    return absF(a - b) <= tol;
}

static bool vetorProximo(const float* a, const float* b, int n, float tol = 1e-4f) {
    for(int i = 0; i < n; i++)
        if(!proximos(a[i], b[i], tol)) return false;
    return true;
}

static bool tudoFinito(const float* v, int n) {
    for(int i = 0; i < n; i++)
        if(!isfinite(v[i])) return false;
    return true;
}

// === verificação de gradiente por diferenças finitas ===
// retorna o erro relativo maximo entre gradiente analitico e numerico
static float verificarGradienteEntrada(Densa& d, const float* x, const float* dL_dy, float eps = 1e-3f) {
    int nE = d.nEnt, nS = d.nSai;
    float* y  = (float*)malloc(nS * sizeof(float));
    float* y2 = (float*)malloc(nS * sizeof(float));
    float* xp = (float*)malloc(nE * sizeof(float));
    float* gradAnal = (float*)calloc(nE, sizeof(float));

    // gradiente analitico
    d.prop(x, y);
    d.zerarGrad();
    d.retroprop(dL_dy, gradAnal);

    float errMax = 0.0f;
    for(int i = 0; i < nE; i++) {
        memcpy(xp, x, nE * sizeof(float));
        xp[i] += eps; d.prop(xp, y);
        float lp = 0; for(int o = 0; o < nS; o++) lp += dL_dy[o] * y[o];

        xp[i] -= 2*eps; d.prop(xp, y2);
        float lm = 0; for(int o = 0; o < nS; o++) lm += dL_dy[o] * y2[o];

        float gNum = (lp - lm) / (2 * eps);
        float denom = absF(gNum) > 1e-8f ? absF(gNum) : 1e-8f;
        float err = absF(gradAnal[i] - gNum) / denom;
        if(err > errMax) errMax = err;
    }
    free(y);
    free(y2);
    free(xp);
    free(gradAnal);
    return errMax;
}

static float verificarGradientePesos(Densa& d, const float* x, const float* dL_dy, float eps = 1e-3f) {
    int nE = d.nEnt, nS = d.nSai;
    float* y = (float*)malloc(nS * sizeof(float));
    float* y2 = (float*)malloc(nS * sizeof(float));
    float errMax = 0.0f;

    d.prop(x, y);
    d.zerarGrad();
    d.retroprop(dL_dy, nullptr);

    for(int idc = 0; idc < nS * nE; idc++) {
        float orig = d.pesos[idc];

        d.pesos[idc] = orig + eps; d.prop(x, y);
        float lp = 0; for(int o = 0; o < nS; o++) lp += dL_dy[o] * y[o];

        d.pesos[idc] = orig - eps; d.prop(x, y2);
        float lm = 0; for(int o = 0; o < nS; o++) lm += dL_dy[o] * y2[o];

        d.pesos[idc] = orig;

        float gNum = (lp - lm) / (2 * eps);
        float denom = absF(gNum) > 1e-8f ? absF(gNum) : 1e-8f;
        float err = absF(d.gradP[idc] - gNum) / denom;
        if(err > errMax) errMax = err;
    }
    free(y); free(y2);
    return errMax;
}

// TESTES
void testePropLinear() {
    printf("\n[1] Propagação sem ativacao (linear)\n");
    // 2 entradas, 2 saidas, sem ativação
    Densa d(2, 2, "");
    // pesos: [[1,2],[3,4]], bias: [0.5, -0.5]
    d.pesos[0]=1; d.pesos[1]=2;
    d.pesos[2]=3; d.pesos[3]=4;
    d.bias[0]=0.5f; d.bias[1]=-0.5f;

    float entrada[2] = {1.0f, 1.0f};
    float saida[2];
    d.prop(entrada, saida);

    // saida[0] = 1*1 + 2*1 + 0.5 = 3.5
    // saida[1] = 3*1 + 4*1 - 0.5 = 6.5
    check("saida[0] == 3.5", proximos(saida[0], 3.5f));
    check("saida[1] == 6.5", proximos(saida[1], 6.5f));
}

void testePropReLU() {
    printf("\n[2] Propagação com ReLU\n");
    Densa d(2, 2, "relu");
    d.pesos[0]=1; d.pesos[1]=-1;
    d.pesos[2]=-1; d.pesos[3]=1;
    d.bias[0]=0; d.bias[1]=0;

    float entrada[2] = {2.0f, 1.0f};
    float saida[2];
    d.prop(entrada, saida);

    // pre[0] = 2-1 = 1 -> relu = 1
    // pre[1] = -2+1 = -1 -> relu = 0
    check("saida[0] == 1 (relu positivo)", proximos(saida[0], 1.0f));
    check("saida[1] == 0 (relu negativo zerado)", proximos(saida[1], 0.0f));
}

void testePropSigmoid() {
    printf("\n[3] Propagação com sigmoid\n");
    Densa d(1, 1, "sigmoid");
    d.pesos[0] = 0.0f;
    d.bias[0] = 0.0f;
    float entrada[1] = {0.0f};
    float saida[1];
    d.prop(entrada, saida);
    // sigmoid(0) = 0.5
    check("sigmoid(0) == 0.5", proximos(saida[0], 0.5f));
}

void testeGradienteAnaliticoVsNumerico() {
    printf("\n[4] Gradiente analitico vs numerico (relu, 4->3)\n");
    srand(42);
    Densa d(4, 3, "relu");

    float x[4] = {0.5f, -0.3f, 1.2f, -0.7f};
    float dLdy[3] = {1.0f, -1.0f, 0.5f};

    float errEnt = verificarGradienteEntrada(d, x, dLdy);
    float errPeso = verificarGradientePesos(d, x, dLdy);

    check("erro gradiente entrada < 1%",  errEnt  < 0.01f);
    check("erro gradiente pesos  < 1%",  errPeso < 0.01f);
    printf("     (errEnt=%.2e, errPeso=%.2e)\n", errEnt, errPeso);
}

void testeGradienteMultiplasAtivacoes() {
    printf("\n[5] Gradiente numerico — multiplas ativacoes (3->2)\n");
    const char* ativs[] = {
        "sigmoid","tanh","swish",
        "gelu","elu","mish","leakyrelu"
    };
    srand(7);

    float x[3] = {0.4f, -0.6f, 1.1f};
    float dLdy[2] = {1.0f, -0.5f};

    for(int a = 0; a < 7; a++) {
        Densa d(3, 2, ativs[a]);
        float err = verificarGradienteEntrada(d, x, dLdy);
        char nome[64];
        snprintf(nome, 64, "%s err < 1%%", ativs[a]);
        check(nome, err < 0.01f);
    }
}

void testeAcumulacaoGradienteBatch() {
    printf("\n[6] Acumulacao de gradiente em batch\n");
    srand(1);
    Densa d(2, 2, "relu");
    d.pesos[0]=0.5f; d.pesos[1]=0.5f;
    d.pesos[2]=0.5f; d.pesos[3]=0.5f;
    d.bias[0]=0; d.bias[1]=0;

    float x1[2] = {1.0f, 0.0f}, x2[2] = {0.0f, 1.0f};
    float dLdy[2] = {1.0f, 1.0f};
    float y[2], ge[2];

    d.zerarGrad();
    d.prop(x1, y); d.retroprop(dLdy, ge);
    d.prop(x2, y); d.retroprop(dLdy, ge);

    // gradP[0*2+0] = delta[0]*x1[0] + delta[0]*x2[0] = 1*1 + 1*0 = 1
    // gradP[0*2+1] = delta[0]*x1[1] + delta[0]*x2[1] = 1*0 + 1*1 = 1
    // (relu ativada em todos porque pre = 0.5 > 0)
    check("gradP acumulado[0,0] == 1", proximos(d.gradP[0], 1.0f));
    check("gradP acumulado[0,1] == 1", proximos(d.gradP[1], 1.0f));
    check("zerarGrad funciona", [&]{
        d.zerarGrad();
        return proximos(d.gradP[0], 0.0f);
    }());
}

void testeSGDConverge() {
    printf("\n[7] SGD converge em problema XOR (tanh, 2->4->1)\n");
    srand(123);
    // camadas: entrada(2) -> oculta(4,tanh) -> saida(1,sigmoid)
    Densa oculta(2, 4, "tanh");
    Densa saida(4, 1, "sigmoid");

    float xor_x[4][2] = {{0,0},{0,1},{1,0},{1,1}};
    float xor_y[4]    = {0,1,1,0};

    float lr = 0.1f;
    float perdaInicial = 0.0f, perdaFinal = 0.0f;

    // calcula perda inicial
    for(int i = 0; i < 4; i++) {
        float h[4], out[1];
        oculta.prop(xor_x[i], h);
        saida.prop(h, out);
        float diff = out[0] - xor_y[i];
        perdaInicial += diff * diff;
    }
    perdaInicial /= 4;

    // treina 10000 épocas
    for(int ep = 0; ep < 10000; ep++) {
        for(int i = 0; i < 4; i++) {
            float h[4], out[1];
            oculta.prop(xor_x[i], h);
            saida.prop(h, out);

            float gradSai[1] = { 2.0f * (out[0] - xor_y[i]) };
            float gradMeio[4];

            oculta.zerarGrad(); saida.zerarGrad();
            saida.retroprop(gradSai, gradMeio);
            oculta.retroprop(gradMeio, nullptr);

            saida.cliparGrad(5.0f);
            oculta.cliparGrad(5.0f);

            saida.atualizarSGD(lr);
            oculta.atualizarSGD(lr);
        }
    }
    for(int i = 0; i < 4; i++) {
        float h[4], out[1];
        oculta.prop(xor_x[i], h);
        saida.prop(h, out);
        float diff = out[0] - xor_y[i];
        perdaFinal += diff * diff;
    }
    perdaFinal /= 4;

    printf("     perda inicial=%.4f  perda final=%.4f\n", perdaInicial, perdaFinal);
    check("perda final < 0.01", perdaFinal < 0.01f);
}

void testeEntradaZerada() {
    printf("\n[8] Entrada zerada\n");
    srand(5);
    Densa d(3, 3, "relu");
    float x[3] = {0,0,0}, y[3];
    d.prop(x, y);
    // saida = relu(bias), bias=0 → tudo zero
    check("saida toda zero com entrada e bias zeros", proximos(y[0],0) && proximos(y[1],0) && proximos(y[2],0));
}

void testeEntradaGrande() {
    printf("\n[9] Entrada muito grande (saturacao sigmoid)\n");
    Densa d(1, 1, "sigmoid");
    d.pesos[0] = 1.0f; d.bias[0] = 0.0f;

    float xGrande[1] = {100.0f}, yG[1];
    float xPeq[1] = {-100.0f}, yP[1];
    d.prop(xGrande, yG);
    d.prop(xPeq, yP);

    check("sigmoid(100) aprox 1", proximos(yG[0], 1.0f, 1e-3f));
    check("sigmoid(-100) aprox 0", proximos(yP[0], 0.0f, 1e-3f));
    check("saidas finitas", isfinite(yG[0]) && isfinite(yP[0]));
}

void testeNaNPropagacao() {
    printf("\n[10] NaN nao deve surgir com pesos normais e entrada normal\n");
    srand(99);
    Densa d(8, 8, "gelu");
    float x[8] = {0.1f,-0.2f,0.3f,-0.4f,0.5f,-0.6f,0.7f,-0.8f};
    float y[8], ge[8];
    float dLdy[8]; for(int i=0;i<8;i++) dLdy[i]=0.1f*(i+1);

    d.prop(x, y);
    check("Propagação sem NaN/Inf", tudoFinito(y, 8));

    d.zerarGrad();
    d.retroprop(dLdy, ge);
    check("gradEntrada sem NaN/Inf", tudoFinito(ge, 8));
    check("gradPesos sem NaN/Inf", tudoFinito(d.gradP, 8*8));
}

void testeClipGrad() {
    printf("\n[11] Clipping de gradiente\n");
    Densa d(2, 2, "relu");
    // define gradientes grandes manualmente
    for(int i=0; i<4; i++) d.gradP[i] = 100.0f;
    for(int i=0; i<2; i++) d.gradB[i] = 100.0f;

    d.cliparGrad(1.0f);

    float norma2 = 0;
    for(int i=0;i<4;i++) norma2 += d.gradP[i]*d.gradP[i];
    for(int i=0;i<2;i++) norma2 += d.gradB[i]*d.gradB[i];
    float norma = sqrtf(norma2);
    check("norma apos clip <= 1.0 + eps", norma <= 1.0f + 1e-4f);
}

void testeInicializacaoHe() {
    printf("\n[12] Inicializacao He (variancia aproximada 2/nEnt)\n");
    srand(42);
    Densa d(100, 100, "relu");
    d.inicializar("he");

    // calcula variância amostral
    float soma=0, soma2=0;
    int n = 100*100;
    for(int i=0;i<n;i++) {
        soma+=d.pesos[i];
        soma2+=d.pesos[i]*d.pesos[i];
    }
    float media = soma/n;
    float var = soma2/n - media*media;
    // He uniforme: var = (6/nEnt)/3 = 2/nEnt = 0.02, tolerancia generosa
    printf("     variancia amostral = %.4f  (esperado ~%.4f)\n", var, 2.0f/100);
    check("variancia proxima de 2/nEnt (±50%%)", var > 0.01f && var < 0.03f);
}

void testeNumParams() {
    printf("\n[13] numParams correto\n");
    Densa d(5, 3, "relu");
    check("numParams == 5*3 + 3 == 18", d.numParams() == 18);
}

void testeCamadaGrande() {
    printf("\n[14] Camada grande (256->256, relu) sem crash\n");
    srand(7);
    Densa d(256, 256, "relu");
    float* x = (float*)calloc(256, sizeof(float));
    float* y = (float*)calloc(256, sizeof(float));
    float* dLdy = (float*)calloc(256, sizeof(float));
    float* ge = (float*)calloc(256, sizeof(float));
    for(int i=0;i<256;i++) { x[i]=(float)(i%7-3)*0.1f; dLdy[i]=0.01f; }

    d.prop(x, y);
    d.zerarGrad();
    d.retroprop(dLdy, ge);

    check("Propagação 256->256 finito", tudoFinito(y, 256));
    check("gradEntrada 256->256 finito", tudoFinito(ge, 256));
    check("gradPesos 256->256 finito", tudoFinito(d.gradP, 256*256));

    free(x);
    free(y);
    free(dLdy);
    free(ge);
}

void testeAtivacaoNula() {
    printf("\n[15] Camada linear (sem ativacao)\n");
    Densa d(2, 2, ""); // ativação vazia = identidade
    d.pesos[0]=2; d.pesos[1]=0;
    d.pesos[2]=0; d.pesos[3]=3;
    d.bias[0]=1; d.bias[1]=-1;
    float x[2]={1,1}, y[2];
    d.prop(x, y);
    check("saida[0] == 2+0+1 == 3", proximos(y[0], 3.0f));
    check("saida[1] == 0+3-1 == 2", proximos(y[1], 2.0f));
}

int main() {
    printf("=== TESTES Densa ===\n");

    testePropLinear();
    testePropReLU();
    testePropSigmoid();
    testeGradienteAnaliticoVsNumerico();
    testeGradienteMultiplasAtivacoes();
    testeAcumulacaoGradienteBatch();
    testeSGDConverge();
    testeEntradaZerada();
    testeEntradaGrande();
    testeNaNPropagacao();
    testeClipGrad();
    testeInicializacaoHe();
    testeNumParams();
    testeCamadaGrande();
    testeAtivacaoNula();

    printf("\n============================\n");
    printf("Resultado: %d/%d passaram\n", _passou, _total);
    return _passou == _total ? 0 : 1;
}