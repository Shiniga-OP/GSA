// biblis/otimis/sgd.h
#pragma once
#include "otimizador.h"

// SGD com Momentum, Nesterov opcional
// buffer unico: velocidade[totalN]
// Nesterov: olha pro futuro no gradiente antes de att velocidade
// decaimento peso: L2 no gradiente(formulação SGD classica)
struct SGD {
    float taxa;
    float momentum;
    float pd;
    bool nesterov;
    int totalN;

    float* vel; // velocidade contigua

    static const int MAX_GRUPOS = 64;
    float* pPtrs[MAX_GRUPOS];
    float* gPtrs[MAX_GRUPOS];
    int pTams[MAX_GRUPOS];
    int gTams[MAX_GRUPOS];
    int nGrupos;

    void iniciar(Camada** camadas, int nCamadas,
    float taxa_ = 1e-2f, float momentum_ = 0.9f,
    float pd_ = 0.0f, bool nesterov_ = false) {
        taxa = taxa_;
        momentum = momentum_;
        pd = pd_;
        nesterov = nesterov_;
        totalN = _totalParams(camadas, nCamadas);
        vel = (float*)calloc(totalN, sizeof(float));
        nGrupos = _coletarPtrs(camadas, nCamadas,
        pPtrs, pTams, gPtrs, gTams, MAX_GRUPOS);
    }

    void liberar() { free(vel); vel = nullptr; }

    void att() {
        int pos = 0;
        for(int g = 0; g < nGrupos; g++) {
            float* __restrict__ p  = pPtrs[g];
            float* __restrict__ gd = gPtrs[g];
            float* __restrict__ vg = vel + pos;
            int n = pTams[g];
            int k = 0;

            if(!nesterov) {
                // SGD padrão com momentum
                for(; k <= n - 8; k += 8) {
                    // g_eff = g + pd*p
                    float ge0 = gd[k]   + pd*p[k];
                    float ge1 = gd[k+1] + pd*p[k+1];
                    float ge2 = gd[k+2] + pd*p[k+2];
                    float ge3 = gd[k+3] + pd*p[k+3];
                    float ge4 = gd[k+4] + pd*p[k+4];
                    float ge5 = gd[k+5] + pd*p[k+5];
                    float ge6 = gd[k+6] + pd*p[k+6];
                    float ge7 = gd[k+7] + pd*p[k+7];
                    // v = momentum*v + g_eff
                    vg[k] = momentum*vg[k] + ge0;
                    vg[k+1] = momentum*vg[k+1] + ge1;
                    vg[k+2] = momentum*vg[k+2] + ge2;
                    vg[k+3] = momentum*vg[k+3] + ge3;
                    vg[k+4] = momentum*vg[k+4] + ge4;
                    vg[k+5] = momentum*vg[k+5] + ge5;
                    vg[k+6] = momentum*vg[k+6] + ge6;
                    vg[k+7] = momentum*vg[k+7] + ge7;
                    // p -= taxa * v
                    p[k] -= taxa*vg[k];
                    p[k+1] -= taxa*vg[k+1];
                    p[k+2] -= taxa*vg[k+2];
                    p[k+3] -= taxa*vg[k+3];
                    p[k+4] -= taxa*vg[k+4];
                    p[k+5] -= taxa*vg[k+5];
                    p[k+6] -= taxa*vg[k+6];
                    p[k+7] -= taxa*vg[k+7];
                }
                for(; k < n; k++) {
                    float ge = gd[k] + pd*p[k];
                    vg[k] = momentum*vg[k] + ge;
                    p[k] -= taxa * vg[k];
                }
            } else {
                // nesterov: p -= taxa*(momentum*v + g_eff)
                for(; k <= n - 8; k += 8) {
                    float ge0 = gd[k] + pd*p[k];
                    float ge1 = gd[k+1] + pd*p[k+1];
                    float ge2 = gd[k+2] + pd*p[k+2];
                    float ge3 = gd[k+3] + pd*p[k+3];
                    float ge4 = gd[k+4] + pd*p[k+4];
                    float ge5 = gd[k+5] + pd*p[k+5];
                    float ge6 = gd[k+6] + pd*p[k+6];
                    float ge7 = gd[k+7] + pd*p[k+7];
                    vg[k] = momentum*vg[k] + ge0;
                    vg[k+1] = momentum*vg[k+1] + ge1;
                    vg[k+2] = momentum*vg[k+2] + ge2;
                    vg[k+3] = momentum*vg[k+3] + ge3;
                    vg[k+4] = momentum*vg[k+4] + ge4;
                    vg[k+5] = momentum*vg[k+5] + ge5;
                    vg[k+6] = momentum*vg[k+6] + ge6;
                    vg[k+7] = momentum*vg[k+7] + ge7;
                    p[k] -= taxa*(momentum*vg[k] + ge0);
                    p[k+1] -= taxa*(momentum*vg[k+1] + ge1);
                    p[k+2] -= taxa*(momentum*vg[k+2] + ge2);
                    p[k+3] -= taxa*(momentum*vg[k+3] + ge3);
                    p[k+4] -= taxa*(momentum*vg[k+4] + ge4);
                    p[k+5] -= taxa*(momentum*vg[k+5] + ge5);
                    p[k+6] -= taxa*(momentum*vg[k+6] + ge6);
                    p[k+7] -= taxa*(momentum*vg[k+7] + ge7);
                }
                for(; k < n; k++) {
                    float ge = gd[k] + pd*p[k];
                    vg[k] = momentum*vg[k] + ge;
                    p[k] -= taxa*(momentum*vg[k] + ge);
                }
            }
            pos += n;
        }
    }

    void zerarGrads(Camada** camadas, int nCamadas) {
        for(int c = 0; c < nCamadas; c++) camadas[c]->zerarGrad();
    }
};