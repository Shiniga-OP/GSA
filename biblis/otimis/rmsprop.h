// biblis/rmsprop.h
#pragma once
#include "otimizador.h"

// RMSProp para redes recorrentes/treino não-estacionario
// buffer unico: v[totalN]
// sem momentum de primeira ordem(pode ser adicionado se necessario)
struct RMSProp {
    float taxa;
    float alpha; // decaimento da media(tipicamente 0.99)
    float eps;
    float pd;
    float momentum;
    int totalN;

    float* v; // media quadratica
    float* buf; // buffer de momentum(apenas se momentum > 0)

    static const int MAX_GRUPOS = 64;
    float* pPtrs[MAX_GRUPOS];
    float* gPtrs[MAX_GRUPOS];
    int pTams[MAX_GRUPOS];
    int gTams[MAX_GRUPOS];
    int nGrupos;

    void iniciar(Camada** camadas, int nCamadas,
    float taxa_ = 1e-3f, float alpha_ = 0.99f,
    float eps_ = 1e-8f, float pd_ = 0.0f,
    float momentum_ = 0.0f) {
        taxa = taxa_;
        alpha = alpha_;
        eps = eps_;
        pd = pd_;
        momentum = momentum_;
        totalN = _totalParams(camadas, nCamadas);
        int extra = (momentum > 0.0f) ? 2 : 1;
        v = (float*)calloc(totalN * extra, sizeof(float));
        buf = (momentum > 0.0f) ? v + totalN : nullptr;
        nGrupos = _coletarPtrs(camadas, nCamadas,
        pPtrs, pTams, gPtrs, gTams, MAX_GRUPOS);
    }

    void liberar() { free(v); v = nullptr; buf = nullptr; }

    void att() {
        int pos = 0;
        float oa = 1.0f - alpha;
        for(int g = 0; g < nGrupos; g++) {
            float* __restrict__ p  = pPtrs[g];
            float* __restrict__ gd = gPtrs[g];
            float* __restrict__ vg = v + pos;
            int n = pTams[g];
            int k = 0;

            if(buf == nullptr) {
                // sem momentum
                for(; k <= n - 8; k += 8) {
                    float g0=gd[k]+pd*p[k], g1=gd[k+1]+pd*p[k+1];
                    float g2=gd[k+2]+pd*p[k+2], g3=gd[k+3]+pd*p[k+3];
                    float g4=gd[k+4]+pd*p[k+4], g5=gd[k+5]+pd*p[k+5];
                    float g6=gd[k+6]+pd*p[k+6], g7=gd[k+7]+pd*p[k+7];
                    vg[k] = alpha*vg[k] + oa*g0*g0;
                    vg[k+1] = alpha*vg[k+1] + oa*g1*g1;
                    vg[k+2] = alpha*vg[k+2] + oa*g2*g2;
                    vg[k+3] = alpha*vg[k+3] + oa*g3*g3;
                    vg[k+4] = alpha*vg[k+4] + oa*g4*g4;
                    vg[k+5] = alpha*vg[k+5] + oa*g5*g5;
                    vg[k+6] = alpha*vg[k+6] + oa*g6*g6;
                    vg[k+7] = alpha*vg[k+7] + oa*g7*g7;
                    p[k] -= taxa*g0/(sqrtf(vg[k])  +eps);
                    p[k+1] -= taxa*g1/(sqrtf(vg[k+1])+eps);
                    p[k+2] -= taxa*g2/(sqrtf(vg[k+2])+eps);
                    p[k+3] -= taxa*g3/(sqrtf(vg[k+3])+eps);
                    p[k+4] -= taxa*g4/(sqrtf(vg[k+4])+eps);
                    p[k+5] -= taxa*g5/(sqrtf(vg[k+5])+eps);
                    p[k+6] -= taxa*g6/(sqrtf(vg[k+6])+eps);
                    p[k+7] -= taxa*g7/(sqrtf(vg[k+7])+eps);
                }
                for(; k < n; k++) {
                    float gk = gd[k] + pd*p[k];
                    vg[k] = alpha*vg[k] + oa*gk*gk;
                    p[k] -= taxa*gk/(sqrtf(vg[k])+eps);
                }
            } else {
                float* __restrict__ bg = buf + pos;
                for(; k <= n - 8; k += 8) {
                    float g0=gd[k]+pd*p[k], g1=gd[k+1]+pd*p[k+1];
                    float g2=gd[k+2]+pd*p[k+2], g3=gd[k+3]+pd*p[k+3];
                    float g4=gd[k+4]+pd*p[k+4], g5=gd[k+5]+pd*p[k+5];
                    float g6=gd[k+6]+pd*p[k+6], g7=gd[k+7]+pd*p[k+7];
                    vg[k] = alpha*vg[k] + oa*g0*g0;
                    vg[k+1] = alpha*vg[k+1] + oa*g1*g1;
                    vg[k+2] = alpha*vg[k+2] + oa*g2*g2;
                    vg[k+3] = alpha*vg[k+3] + oa*g3*g3;
                    vg[k+4] = alpha*vg[k+4] + oa*g4*g4;
                    vg[k+5] = alpha*vg[k+5] + oa*g5*g5;
                    vg[k+6] = alpha*vg[k+6] + oa*g6*g6;
                    vg[k+7] = alpha*vg[k+7] + oa*g7*g7;
                    float passo0=taxa*g0/(sqrtf(vg[k])+eps);
                    float passo1=taxa*g1/(sqrtf(vg[k+1])+eps);
                    float passo2=taxa*g2/(sqrtf(vg[k+2])+eps);
                    float passo3=taxa*g3/(sqrtf(vg[k+3])+eps);
                    float passo4=taxa*g4/(sqrtf(vg[k+4])+eps);
                    float passo5=taxa*g5/(sqrtf(vg[k+5])+eps);
                    float passo6=taxa*g6/(sqrtf(vg[k+6])+eps);
                    float passo7=taxa*g7/(sqrtf(vg[k+7])+eps);
                    bg[k]=momentum*bg[k]+passo0;
                    p[k]-=bg[k];
                    bg[k+1]=momentum*bg[k+1]+passo1;
                    p[k+1]-=bg[k+1];
                    bg[k+2]=momentum*bg[k+2]+passo2;
                    p[k+2]-=bg[k+2];
                    bg[k+3]=momentum*bg[k+3]+passo3;
                    p[k+3]-=bg[k+3];
                    bg[k+4]=momentum*bg[k+4]+passo4;
                    p[k+4]-=bg[k+4];
                    bg[k+5]=momentum*bg[k+5]+passo5;
                    p[k+5]-=bg[k+5];
                    bg[k+6]=momentum*bg[k+6]+passo6;
                    p[k+6]-=bg[k+6];
                    bg[k+7]=momentum*bg[k+7]+passo7;
                    p[k+7]-=bg[k+7];
                }
                for(; k < n; k++) {
                    float gk = gd[k] + pd*p[k];
                    vg[k] = alpha*vg[k] + oa*gk*gk;
                    float s = taxa*gk/(sqrtf(vg[k])+eps);
                    bg[k] = momentum*bg[k] + s;
                    p[k] -= bg[k];
                }
            }
            pos += n;
        }
    }

    void zerarGrads(Camada** camadas, int nCamadas) {
        for(int c = 0; c < nCamadas; c++) camadas[c]->zerarGrad();
    }
};