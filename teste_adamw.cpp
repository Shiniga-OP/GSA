// teste_adamw.cpp
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include "biblis/camadas/densa.h"
#include "biblis/otimis/adamw.h"

// utilidades
static int falhas = 0;
static int total  = 0;

static void checar(const char* nome, bool cond) {
    total++;
    if(!cond) {
        falhas++;
        printf("  [FALHA] %s\n", nome);
    } else {
        printf("  [OK]    %s\n", nome);
    }
}

static float absf_(float x) { return x < 0 ? -x : x; }

// teste 1: convergencia em regressão quadratica simples
// Aprende f(x) = x^2 com rede [1->16->1]
// criterio: perda < 0.01 em <= 2000 passos
void teste_convergencia_regressao() {
    printf("\n[TESTE 1] Convergência: regressão x^2\n");

    srand(42);
    Densa c1(1, 16, "relu");
    Densa c2(16, 1, ""); // sem ativação na saida
    Camada* camadas[] = { &c1, &c2 };

    AdamW adam;
    adam.iniciar(camadas, 2, 1e-3f);

    float buf1[16], buf2[1], g2[16], g1[1];
    int convergiu = 0;
    float ultima_perda = 1e9f;

    for(int passo = 0; passo < 2000 && !convergiu; passo++) {
        // mini-lote de 16 amostras
        adam.zerarGrads(camadas, 2);
        float perda_total = 0.0f;

        for(int b = 0; b < 16; b++) {
            float x  = ((float)rand() / RAND_MAXf) * 4.0f - 2.0f; // [-2, 2]
            float alvo = x * x;

            float entrada[1] = { x };
            c1.prop(entrada, buf1);
            c2.prop(buf1, buf2);

            float err = buf2[0] - alvo;
            perda_total += err * err;

            // retroprop
            float gSaida[1] = { 2.0f * err / 16.0f };
            c2.retroprop(gSaida, g2);
            c1.retroprop(g2, g1);
        }
        ultima_perda = perda_total / 16.0f;
        adam.att();

        if(ultima_perda < 0.01f) convergiu = 1;
    }
    checar("perda < 0.01 em 2000 passos", convergiu);
    checar("perda não é NaN", !isnan(ultima_perda));
    checar("perda não é Inf", !isinf(ultima_perda));

    adam.liberar();
}

// teste 2: decaimento de peso
// os parametros devem diminuir quando gradiente é zero
void teste_decaimento_peso() {
    printf("\n[TESTE 2] Decaimento de peso com gradiente zero\n");

    srand(1);
    Densa c1(4, 4, "relu");
    Camada* camadas[] = { &c1 };

    // força pesos a 1.0
    for(int i = 0; i < 4*4; i++) c1.pesos[i] = 1.0f;
    for(int i = 0; i < 4; i++) c1.bias[i] = 1.0f;

    AdamW adam;
    // taxa alta, pd alto, para efeito visivel
    adam.iniciar(camadas, 1, 1e-2f, 0.9f, 0.999f, 1e-8f, 0.1f);

    // zera gradientes mas roda N passos
    for(int p = 0; p < 100; p++) {
        adam.zerarGrads(camadas, 1);
        adam.att();
    }
    // Com gradiente zero e pd>0, os pesos devem ter decaído
    float soma_pesos = 0.0f;
    for(int i = 0; i < 4*4; i++) soma_pesos += c1.pesos[i];
    float media = soma_pesos / 16.0f;

    checar("pesos decaíram (media < 0.95)", media < 0.95f);
    checar("pesos não zeraram (media > 0.0)", media > 0.0f);

    adam.liberar();
}

// TESTE 3: zerarGrads limpa gradientes de todas as camadas
void teste_zerar_grads() {
    printf("\n[TESTE 3] zerarGrads limpa todos os gradientes\n");

    srand(7);
    Densa c1(8, 8, "relu");
    Densa c2(8, 4, "");
    Camada* camadas[] = { &c1, &c2 };

    AdamW adam;
    adam.iniciar(camadas, 2);

    // propaga e retropropaga para sujar gradientes
    float ent[8], s1[8], s2[4], g1[8], g2[8];
    for(int i = 0; i < 8; i++) ent[i] = 1.0f;
    c1.prop(ent, s1);
    c2.prop(s1, s2);
    float gs2[4];
    for(int i = 0; i < 4; i++) gs2[i] = 1.0f;
    c2.retroprop(gs2, g1);
    c1.retroprop(g1, g2);

    // confirma que gradientes estão sujos
    float norma_antes = 0.0f;
    for(int i = 0; i < 8*8; i++) norma_antes += c1.gradP[i]*c1.gradP[i];

    adam.zerarGrads(camadas, 2);

    float norma_depois = 0.0f;
    for(int i = 0; i < 8*8; i++) norma_depois += c1.gradP[i]*c1.gradP[i];
    for(int i = 0; i < 8*4; i++) norma_depois += c2.gradP[i]*c2.gradP[i];
    for(int i = 0; i < 8; i++) norma_depois += c1.gradB[i]*c1.gradB[i];
    for(int i = 0; i < 4; i++) norma_depois += c2.gradB[i]*c2.gradB[i];

    checar("gradientes estavam sujos antes", norma_antes > 0.0f);
    checar("gradientes zerados após zerarGrads", norma_depois == 0.0f);

    adam.liberar();
}

// teste 4: primeiros passos tem taxa efetiva pequena
// testa que o passo 1 não explode(efeito do beta2 na correção)
void teste_bias_correção() {
    printf("\n[TESTE 4] Bias correção, passo 1 não explode\n");

    srand(3);
    Densa c1(32, 32, "relu");
    Camada* camadas[] = { &c1 };

    // força pesos a valor conhecido
    for(int i = 0; i < 32*32; i++) c1.pesos[i] = 0.5f;
    float pesos_antes[32*32];
    memcpy(pesos_antes, c1.pesos, 32*32*sizeof(float));

    AdamW adam;
    adam.iniciar(camadas, 1, 1e-3f);

    // propaga uma vez, gradiente não zero
    float ent[32], s[32], gs[32], ge[32];
    for(int i = 0; i < 32; i++) ent[i] = 0.1f;
    c1.prop(ent, s);
    for(int i = 0; i < 32; i++) gs[i] = 1.0f;
    c1.retroprop(gs, ge);
    adam.att();

    // calcula maxima variação nos pesos no passo 1
    float max_delta = 0.0f;
    for(int i = 0; i < 32*32; i++) {
        float d = absf_(c1.pesos[i] - pesos_antes[i]);
        if(d > max_delta) max_delta = d;
    }
    // com taxa=1e-3, variação maxima deve ser << 1.0
    checar("variação máxima passo 1 < 0.1", max_delta < 0.1f);
    checar("variação máxima passo 1 > 0", max_delta > 0.0f);

    adam.liberar();
}

// teste 5: contagem de parametros, totalN bate com numParams()
void teste_contagem_params() {
    printf("\n[TESTE 5] Contagem de parâmetros (totalN)\n");

    Densa c1(10, 20, "relu");
    Densa c2(20, 5,  "");
    Camada* camadas[] = { &c1, &c2 };

    AdamW adam;
    adam.iniciar(camadas, 2);

    int esperado = c1.numParams() + c2.numParams(); // 10*20+20 + 20*5+5 = 325
    checar("totalN == soma numParams()", adam.totalN == esperado);
    // c1: 200+20=220, c2: 100+5=105 → 325
    checar("totalN == 325", adam.totalN == 325);

    adam.liberar();
}

// teste 6: classificação XOR, convergencia em problema não-linear
// rede [2->8->1], saida sigmoid, BCE perda, <= 3000 passos
void teste_xor() {
    printf("\n[TESTE 6] Convergência: XOR (problema não-linear)\n");

    srand(99);
    Densa c1(2, 8, "relu");
    Densa c2(8, 1, "sigmoid");
    Camada* camadas[] = { &c1, &c2 };

    AdamW adam;
    adam.iniciar(camadas, 2, 5e-3f);

    float xor_x[4][2] = {{0,0},{0,1},{1,0},{1,1}};
    float xor_y[4] = {0,1,1,0};

    float buf1[8], buf2[1], g2[8], g1[2];
    int convergiu = 0;
    float ultima_acc = 0.0f;

    for(int ep = 0; ep < 3000 && !convergiu; ep++) {
        adam.zerarGrads(camadas, 2);

        for(int b = 0; b < 4; b++) {
            c1.prop(xor_x[b], buf1);
            c2.prop(buf1, buf2);

            float alvo = xor_y[b];
            float pred  = buf2[0];
            // BCE grad: (pred - alvo) / 4
            float gSaida[1] = { (pred - alvo) / 4.0f };
            c2.retroprop(gSaida, g2);
            c1.retroprop(g2, g1);
        }
        adam.att();

        // verifica acuracia
        if(ep % 100 == 0) {
            int acertos = 0;
            for(int b = 0; b < 4; b++) {
                c1.prop(xor_x[b], buf1);
                c2.prop(buf1, buf2);
                int pred = buf2[0] > 0.5f ? 1 : 0;
                if(pred == (int)xor_y[b]) acertos++;
            }
            ultima_acc = acertos / 4.0f;
            if(ultima_acc == 1.0f) convergiu = 1;
        }
    }
    checar("acurácia 100% em XOR em 3000 épocas", convergiu);

    adam.liberar();
}

// teste 7: estabilidade numerica, gradientes grandes(clip manual)
void teste_estabilidade_gradiente_grande() {
    printf("\n[TESTE 7] Estabilidade com gradientes grandes\n");

    srand(55);
    Densa c1(16, 16, "relu");
    Camada* camadas[] = { &c1 };

    AdamW adam;
    adam.iniciar(camadas, 1, 1e-3f);

    float ent[16], s[16], gs[16], ge[16];
    for(int i = 0; i < 16; i++) ent[i] = 1.0f;

    int tem_nan = 0, tem_inf = 0;
    for(int p = 0; p < 100; p++) {
        adam.zerarGrads(camadas, 1);
        c1.prop(ent, s);
        // gradiente grande deliberadamente
        for(int i = 0; i < 16; i++) gs[i] = 1000.0f;
        c1.retroprop(gs, ge);
        adam.att();

        for(int i = 0; i < 16*16; i++) {
            if(isnan(c1.pesos[i])) tem_nan = 1;
            if(isinf(c1.pesos[i])) tem_inf = 1;
        }
    }
    checar("sem NaN após gradientes grandes", !tem_nan);
    checar("sem Inf após gradientes grandes", !tem_inf);

    adam.liberar();
}

// teste 8: Agendador cosseno, valores nos extremos
void teste_agendador_cosseno() {
    printf("\n[TESTE 8] AgendadorCosseno, valores limites\n");

    AgendadorCosseno ag;
    ag.taxaMax = 1e-3f;
    ag.taxaMin = 1e-5f;
    ag.passosTotal = 1000;
    ag.aquecimento = 100;

    float t0 = ag.calcular(0);
    float t50 = ag.calcular(50); // meio do aquecimento
    float t100 = ag.calcular(100); // fim do aquecimento = taxaMax
    float t550 = ag.calcular(550); // meio do cosseno ≈ media
    float t1000= ag.calcular(1000); // final ≈ taxaMin

    checar("t=0 -> taxa = 0", t0 == 0.0f);
    checar("t=50 -> taxa entre 0 e taxaMax", t50 > 0.0f && t50 < ag.taxaMax);
    checar("t=100 -> taxa ≈ taxaMax", absf_(t100 - ag.taxaMax) < 1e-6f);
    checar("t=550 -> taxa entre min e max", t550 > ag.taxaMin && t550 < ag.taxaMax);
    checar("t=1000 -> taxa ≈ taxaMin", absf_(t1000 - ag.taxaMin) < 1e-6f);
}

// teste 9: nGrupos bate com estrutura das camadas
void teste_n_grupos() {
    printf("\n[TESTE 9] nGrupos coletados corretamente\n");

    Densa c1(5, 5, "relu");
    Densa c2(5, 3, "");
    Camada* camadas[] = { &c1, &c2 };

    AdamW adam;
    adam.iniciar(camadas, 2);

    // cada Densa tem 2 grupos (pesos + bias) -> esperado = 4
    checar("nGrupos == 4 para 2 camadas Densa", adam.nGrupos == 4);

    adam.liberar();
}

// teste 10: passo 0 nunca chamado(passo começa em 1 apos att())
void teste_contador_passo() {
    printf("\n[TESTE 10] Contador de passo\n");

    Densa c1(2, 2, "relu");
    Camada* camadas[] = { &c1 };
    AdamW adam;
    adam.iniciar(camadas, 1);

    checar("passo inicial == 0", adam.passo == 0);
    adam.zerarGrads(camadas, 1);
    adam.att();
    checar("passo após 1 att() == 1", adam.passo == 1);
    adam.att();
    adam.att();
    checar("passo após 3 att() == 3", adam.passo == 3);

    adam.liberar();
}

// teste 11: convergencia numerica, minimo de parabola escalar
// perda = (w + b)^2 com entrada=1, alvo=0
// minimo: p + b = 0. Verifica que saida converge para 0,
// não que p isolado vai para 0(bias é livre)
void teste_minimo_parabola() {
    printf("\n[TESTE 11] Convergência numérica: mínimo de parábola\n");

    srand(77);
    Densa c1(1, 1, "");
    c1.pesos[0] = 5.0f;
    c1.bias[0]  = 0.0f;
    Camada* camadas[] = { &c1 };

    AdamW adam;
    adam.iniciar(camadas, 1, 1e-2f, 0.9f, 0.999f, 1e-8f, 0.0f); // pd=0

    float ent[1] = {1.0f}, s[1], ge[1];

    for(int p = 0; p < 5000; p++) {
        adam.zerarGrads(camadas, 1);
        c1.prop(ent, s);
        // perda = s[0]^2, dL/ds = 2*s[0]
        float gs[1] = { 2.0f * s[0] };
        c1.retroprop(gs, ge);
        adam.att();
    }
    // minimo real: w + b = 0, i.e., sa8da ≈ 0
    c1.prop(ent, s);
    checar("saida converge para ~0 (|saida| < 0.01)", absf_(s[0]) < 0.01f);
    checar("p+b ≈ 0", absf_(c1.pesos[0] + c1.bias[0]) < 0.01f);

    adam.liberar();
}

int main() {
    printf("======================================\n");
    printf("  TESTES INTENSIVOS: AdamW\n");
    printf("======================================\n");

    teste_convergencia_regressao();
    teste_decaimento_peso();
    teste_zerar_grads();
    teste_bias_correção();
    teste_contagem_params();
    teste_xor();
    teste_estabilidade_gradiente_grande();
    teste_agendador_cosseno();
    teste_n_grupos();
    teste_contador_passo();
    teste_minimo_parabola();

    printf("\n======================================\n");
    printf("  RESULTADO: %d/%d passaram\n", total - falhas, total);
    if(falhas == 0)
        printf("  TUDO OK\n");
    else
        printf("  %d FALHA(S)\n", falhas);
    printf("======================================\n");

    return falhas > 0 ? 1 : 0;
}