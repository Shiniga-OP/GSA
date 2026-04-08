// biblis/otimis/otimizador.h
#pragma once
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "../camadas/camada.h"

// utilitários internos
// numero total de parametros em um vetor de camadas
static inline int _totalParams(Camada** camadas, int nCamadas) {
    int total = 0;
    for(int c = 0; c < nCamadas; c++) total += camadas[c]->numParams();
    return total;
}

// coleta todos os ponteiros de params/grads em arrays planos
// retorna numero de grupos(pares ptr+tam)
// maxGrupos deve ser >= 2*nCamadas (cada camada tem no maximo 2 grupos: pesos, bias)
static inline int _coletarPtrs(
    Camada** camadas, int nCamadas,
    float** pPtrs, int* pTams,
    float** gPtrs, int* gTams,
    int maxGrupos
) {
    int ng = 0;
    for(int c = 0; c < nCamadas && ng < maxGrupos; c++) {
        int np = camadas[c]->numParams();
        if (np == 0) continue;
        // cada camada expõe no máximo 2 grupos(pesos, bias)
        float* tmpP[2]; int tmpTP[2];
        float* tmpG[2]; int tmpTG[2];
        camadas[c]->params(tmpP, tmpTP);
        camadas[c]->gradParams(tmpG, tmpTG);
        // quantos grupos esta camada tem?
        // numParams() = soma dos tams, descobre quantos grupos contando
        int soma = 0, gi = 0;
        for(gi = 0; gi < 2 && soma < np; gi++) {
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

// agendadores de taxa de aprendizado
struct AgendadorCosseno {
    float taxaMax;
    float taxaMin;
    int passosTotal;
    int aquecimento;

    float calcular(int passo) const {
        if(passo < aquecimento) {
            // linear aquecimento
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
    float fator; // multiplica taxa a cada 'passo'
    int intervalo; // passos entre cada redução

    float calcular(int passo) const {
        int n = passo / (intervalo > 0 ? intervalo : 1);
        float taxa = taxaInicial;
        for(int i = 0; i < n; i++) taxa *= fator;
        return taxa;
    }
};