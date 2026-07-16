// teste_modelo.cpp
// demonstra o uso completo de Modelo: construcao, propagação, perda, retropropagação,
// um passo de otimizacao com AdamW, e geracao gulosa.
#include <cstdio>
#include <cmath>
#include "biblis/modelo.h"
#include "biblis/otimis/adamw.h"

int main() {
    // dimensoes pequenas de propósito, so pra validar o encadeamento
    int vocab = 50;
    int dim = 16;
    int nCab = 4;
    int dimFF = 64;
    int nCamadas = 2;
    int seqMax = 8;
    int seq = 6;

    printf("=== teste_modelo ===\n");
    printf("vocab=%d dim=%d nCab=%d dimFF=%d nCamadas=%d seqMax=%d seq=%d\n",
           vocab, dim, nCab, dimFF, nCamadas, seqMax, seq);

    // --- 1. construcao ---
    Modelo modelo(vocab, dim, nCab, dimFF, nCamadas, seqMax);
    modelo.inicializar("xavier");

    // total de camadas pro otimizador: 1 (embedding) + nCamadas (blocos) + 1 (saida)
    printf("totalCamadas=%d (esperado %d)\n", modelo.totalCamadas, 1 + nCamadas + 1);

    // --- 2. otimizador ---
    AdamW otim;
    otim.iniciar(modelo.todasCamadas, modelo.totalCamadas, 3e-4f);
    printf("otimizador: totalN=%d parametros, nGrupos=%d\n", otim.totalN, otim.nGrupos);

    // --- 3. dados sinteticos: ids aleatorios simulando uma janela de treino ---
    // (entrada = ids[0..seq-1], alvo = ids deslocado 1 posicao, como no fabrica_dados.h)
    int idsCompletos[7] = {3, 17, 8, 42, 1, 9, 23}; // seq+1 = 7 ids
    int entrada[6];
    int alvo[6];
    for(int i = 0; i < seq; i++) {
        entrada[i] = idsCompletos[i];
        alvo[i] = idsCompletos[i + 1];
    }
    // --- 4. um passo de treino completo: propagação -> perda -> retropropagação -> update ---
    modelo.defSeq(seq);

    float perdaAntes = -1.0f;
    for(int passo = 0; passo < 5; passo++) {
        modelo.zerarGrad();
        modelo.prop(entrada);
        float perda = modelo.perdaCrossEntropy(alvo);
        modelo.retroprop();
        otim.att();

        printf("passo %d: perda = %f\n", passo, perda);
        if(passo == 0) perdaAntes = perda;
    }

    modelo.zerarGrad();
    modelo.prop(entrada);
    float perdaDepois = modelo.perdaCrossEntropy(alvo);

    bool perdaCaiu = perdaDepois < perdaAntes;
    bool perdaFinita = std::isfinite(perdaDepois);
    printf("\nperda inicial=%f, perda apos 5 passos=%f\n", perdaAntes, perdaDepois);
    printf("perda caiu? %s | perda finita? %s\n",
           perdaCaiu ? "sim (OK)" : "nao (FALHOU)",
           perdaFinita ? "sim (OK)" : "nao (FALHOU)");

    // --- 5. geracao ---
    int msg[3] = {3, 17, 8};
    int gerados[5];
    modelo.gerarGuloso(msg, 3, gerados, 5);
    printf("\ngeracao a partir de [3,17,8]: ");
    for(int i = 0; i < 5; i++) printf("%d ", gerados[i]);
    printf("\n");

    bool ok = perdaCaiu && perdaFinita;
    printf("\nRESULTADO: %s\n", ok ? "PASSOU" : "FALHOU");

    otim.liberar();
    return ok ? 0 : 1;
}