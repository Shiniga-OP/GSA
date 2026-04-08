// biblis/otimis/lion.h
#pragma once
#include "otimizador.h"

// Lion, Sign SGD com momentum(Evo-opt, 2023)
// mais rapido que AdamW: sem sqrt, sem divisão, sem v
// apenas uma variavel de estado(m), atualização = sign(m+g)
// uso de memoria: metade do AdamW
// resultado empirico: converge mais rapido em LLMs grandes
// formulação: atualização = sign(beta1*m + (1-beta1)*g)
// m = beta2*m + (1-beta2)*g
// p -= taxa * (atualização + pd*p)
struct Lion {
    float taxa;
    float beta1; // peso de m no calculo da atualização(tipicamente 0.9)
    float beta2; // decaimento de m(tipicamente 0.99)
    float pd;
    int totalN;

    float* m; // momento unico, buffer contiguo

    static const int MAX_GRUPOS = 64;
    float* pPtrs[MAX_GRUPOS];
    float* gPtrs[MAX_GRUPOS];
    int pTams[MAX_GRUPOS];
    int gTams[MAX_GRUPOS];
    int nGrupos;

    void iniciar(Camada** camadas, int nCamadas,
    float taxa_ = 1e-4f, float beta1_ = 0.9f,
    float beta2_ = 0.99f, float pd_ = 1e-2f) {
        taxa = taxa_;
        beta1 = beta1_;
        beta2 = beta2_;
        pd = pd_;
        totalN = _totalParams(camadas, nCamadas);
        m = (float*)calloc(totalN, sizeof(float));
        nGrupos = _coletarPtrs(camadas, nCamadas,
        pPtrs, pTams, gPtrs, gTams, MAX_GRUPOS);
    }

    void liberar() { free(m); m = nullptr; }

    // sign sem laço: (x > 0) - (x < 0)
    static inline float _sign(float x) {
        return (float)((x > 0.0f) - (x < 0.0f));
    }

    void att() {
        int pos = 0;
        float ob1 = 1.0f - beta1;
        float ob2 = 1.0f - beta2;
        for(int g = 0; g < nGrupos; g++) {
            float* __restrict__ p = pPtrs[g];
            float* __restrict__ gd = gPtrs[g];
            float* __restrict__ mg = m + pos;
            int n = pTams[g];
            int k = 0;

            for(; k <= n - 8; k += 8) {
                float g0=gd[k],  g1=gd[k+1], g2=gd[k+2], g3=gd[k+3];
                float g4=gd[k+4],g5=gd[k+5], g6=gd[k+6], g7=gd[k+7];

                // atualização = sign(beta1*m + (1-beta1)*g)
                float u0 = _sign(beta1*mg[k] + ob1*g0);
                float u1 = _sign(beta1*mg[k+1] + ob1*g1);
                float u2 = _sign(beta1*mg[k+2] + ob1*g2);
                float u3 = _sign(beta1*mg[k+3] + ob1*g3);
                float u4 = _sign(beta1*mg[k+4] + ob1*g4);
                float u5 = _sign(beta1*mg[k+5] + ob1*g5);
                float u6 = _sign(beta1*mg[k+6] + ob1*g6);
                float u7 = _sign(beta1*mg[k+7] + ob1*g7);

                // m = beta2*m + (1-beta2)*g
                mg[k] = beta2*mg[k] + ob2*g0;
                mg[k+1] = beta2*mg[k+1] + ob2*g1;
                mg[k+2] = beta2*mg[k+2] + ob2*g2;
                mg[k+3] = beta2*mg[k+3] + ob2*g3;
                mg[k+4] = beta2*mg[k+4] + ob2*g4;
                mg[k+5] = beta2*mg[k+5] + ob2*g5;
                mg[k+6] = beta2*mg[k+6] + ob2*g6;
                mg[k+7] = beta2*mg[k+7] + ob2*g7;

                // p -= taxa * (u + pd*p)
                p[k] -= taxa*(u0 + pd*p[k]);
                p[k+1] -= taxa*(u1 + pd*p[k+1]);
                p[k+2] -= taxa*(u2 + pd*p[k+2]);
                p[k+3] -= taxa*(u3 + pd*p[k+3]);
                p[k+4] -= taxa*(u4 + pd*p[k+4]);
                p[k+5] -= taxa*(u5 + pd*p[k+5]);
                p[k+6] -= taxa*(u6 + pd*p[k+6]);
                p[k+7] -= taxa*(u7 + pd*p[k+7]);
            }
            for(; k < n; k++) {
                float gk = gd[k];
                float u  = _sign(beta1*mg[k] + ob1*gk);
                mg[k] = beta2*mg[k] + ob2*gk;
                p[k] -= taxa*(u + pd*p[k]);
            }
            pos += n;
        }
    }

    void zerarGrads(Camada** camadas, int nCamadas) {
        for(int c = 0; c < nCamadas; c++) camadas[c]->zerarGrad();
    }
};