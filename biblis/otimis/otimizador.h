// biblis/otimis/otimizador.h
#pragma once
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "../camadas/camada.h"

static inline int _totalParams(Camada** camadas, int nCamadas) {
    int total = 0;
    for(int c = 0; c < nCamadas; c++) total += camadas[c]->numParams();
    return total;
}

static inline int _coletarPtrs(
    Camada** camadas, int nCamadas,
    float** pPtrs, int* pTams,
    float** gPtrs, int* gTams,
    int maxGrupos
) {
    int ng = 0;
    for(int c = 0; c < nCamadas && ng < maxGrupos; c++) {
        int np = camadas[c]->numParams();
        if(np == 0) continue;
        float* tmpP[16]; int tmpTP[16];
        float* tmpG[16]; int tmpTG[16];
        camadas[c]->params(tmpP, tmpTP);
        camadas[c]->gradParams(tmpG, tmpTG);
        int soma = 0, gi = 0;
        for(gi = 0; gi < camadas[c]->grupos && soma < np; gi++) {
            pPtrs[ng] = tmpP[gi];
            pTams[ng] = tmpTP[gi];
            gPtrs[ng] = tmpG[gi];
            gTams[ng] = tmpTG[gi];
            soma += tmpTP[gi];
            ng++;
        }
    }
    return ng;
}

struct AgendadorCosseno {
    float taxaMax;
    float taxaMin;
    int passosTotal;
    int aquecimento;

    float calcular(int passo) const {
        if(passo < aquecimento) {
            return taxaMax * ((float)passo / (float)(aquecimento > 0 ? aquecimento : 1));
        }
        int s = passo - aquecimento;
        int T = passosTotal - aquecimento;
        float cosVal = cosf((float)s / (float)(T > 0 ? T : 1) * 3.14159265f);
        return taxaMin + 0.5f * (taxaMax - taxaMin) * (1.0f + cosVal);
    }
};

struct AgendadorDegrau {
    float taxaInicial;
    float fator;
    int intervalo;

    float calcular(int passo) const {
        int n = passo / (intervalo > 0 ? intervalo : 1);
        float taxa = taxaInicial;
        for(int i = 0; i < n; i++) taxa *= fator;
        return taxa;
    }
};