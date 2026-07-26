// biblis/otimis/adamw.h
#pragma once
#include "otimizador.h"

// AdamW completamente fundido, buffer unico contiguo
// dimensão do buffer de estado: [m0..mN | v0..vN]
// onde N = total de parametros. Zero fragmentação
// bias correção: calculo exato via powf() uma vez por passo
// loop principal: desenrolamento x8, sem divisão interna
// decaimento de peso: aplicado diretamente ao param(formulação AdamW correta,
// não ao gradiente, equivale ao L2)
struct AdamW {
    float taxa;
    float beta1;
    float beta2;
    float eps;
    float pd; // decaimento peso
    int passo;
    int totalN; // total de parametros

    float* estado; // buffer contiguo [m[totalN] | v[totalN]]
    float* m; // alias: estado
    float* v; // alias: estado + totalN

    // ponteiros e tamanhos dos grupos(coletados uma vez em inicio)
    // maximo 64 grupos(32 camadas × 2)
    static const int MAX_GRUPOS = 256; // era 64: estourava com poucas camadas (12 grupos/bloco transformer)
    float* pPtrs[MAX_GRUPOS];
    float* gPtrs[MAX_GRUPOS];
    int pTams[MAX_GRUPOS];
    int gTams[MAX_GRUPOS];
    int nGrupos;

    void iniciar(Camada** camadas, int nCamadas,
    float taxa_ = 1e-3f, float beta1_ = 0.9f,
    float beta2_ = 0.999f, float eps_ = 1e-8f,
    float pd_ = 1e-2f) {
        taxa = taxa_;
        beta1 = beta1_;
        beta2 = beta2_;
        eps = eps_;
        pd = pd_;
        passo = 0;
        totalN = _totalParams(camadas, nCamadas);
        estado = (float*)calloc(2 * totalN, sizeof(float));
        m = estado;
        v = estado + totalN;
        nGrupos = _coletarPtrs(camadas, nCamadas,
        pPtrs, pTams, gPtrs, gTams, MAX_GRUPOS);
    }

    void liberar() {
        free(estado);
        estado = nullptr;
    }

    void att() {
        passo++;

        // clipping de norma POR GRUPO (nao mais global): medido com dados reais
        // (diag_grad.cpp), a Densa final (dim->vocab) tem norma de gradiente
        // consistentemente 2-5x maior que embedding/atencao/FFN durante o
        // treino inteiro. com clip GLOBAL, essa unica camada dominava a norma
        // total e a escala resultante sufocava o aprendizado das outras
        // camadas (que ja tinham gradiente bem menor), causando colapso de
        // modo (a rede aprendia so a prever o token mais frequente). clipando
        // cada grupo pela sua propria norma, nenhuma camada rouba a escala
        // de atualizacao das demais.
        const float normaMax = 1.0f;
        float escalasClip[MAX_GRUPOS];
        for(int g = 0; g < nGrupos; g++) {
            float* gd = gPtrs[g];
            int n = pTams[g];
            double somaSq = 0.0;
            for(int k = 0; k < n; k++) somaSq += (double)gd[k] * (double)gd[k];
            float normaGrupo = sqrtf((float)somaSq);
            escalasClip[g] = (normaGrupo > normaMax) ? (normaMax / (normaGrupo + 1e-6f)) : 1.0f;
        }

        // bias correção: calculado uma vez, fora do loop
        float bc1 = 1.0f - powf(beta1, (float)passo);
        float bc2 = 1.0f - powf(beta2, (float)passo);
        float sqrtBC2 = sqrtf(bc2);
        float taxaCorr = taxa * sqrtBC2 / bc1; // taxa efetivo corrigido

        int pos = 0;
        for(int g = 0; g < nGrupos; g++) {
            float* __restrict__ p  = pPtrs[g];
            float* __restrict__ gd = gPtrs[g];
            float* __restrict__ mg = m + pos;
            float* __restrict__ vg = v + pos;
            int n = pTams[g];
            float escalaClip = escalasClip[g];

            // decaimento peso desacoplado(AdamW): p *= (1 - taxa*pd)
            float pdFator = 1.0f - taxa * pd;
            int k = 0;

            // loop fundido: pd + adam em passo unico, x8
            for(; k <= n - 8; k += 8) {
                // decaimento peso
                p[k] *= pdFator;
                p[k+1] *= pdFator;
                p[k+2] *= pdFator;
                p[k+3] *= pdFator;
                p[k+4] *= pdFator;
                p[k+5] *= pdFator;
                p[k+6] *= pdFator;
                p[k+7] *= pdFator;

                // m = beta1*m + (1-beta1)*g (g escalado pelo clip do proprio grupo)
                float g0=gd[k]*escalaClip, g1=gd[k+1]*escalaClip, g2=gd[k+2]*escalaClip, g3=gd[k+3]*escalaClip;
                float g4=gd[k+4]*escalaClip, g5=gd[k+5]*escalaClip, g6=gd[k+6]*escalaClip, g7=gd[k+7]*escalaClip;
                float ob1 = 1.0f - beta1;
                mg[k] = beta1*mg[k] + ob1*g0;
                mg[k+1] = beta1*mg[k+1] + ob1*g1;
                mg[k+2] = beta1*mg[k+2] + ob1*g2;
                mg[k+3] = beta1*mg[k+3] + ob1*g3;
                mg[k+4] = beta1*mg[k+4] + ob1*g4;
                mg[k+5] = beta1*mg[k+5] + ob1*g5;
                mg[k+6] = beta1*mg[k+6] + ob1*g6;
                mg[k+7] = beta1*mg[k+7] + ob1*g7;

                // v = beta2*v + (1-beta2)*g^2
                float ob2 = 1.0f - beta2;
                vg[k] = beta2*vg[k] + ob2*g0*g0;
                vg[k+1] = beta2*vg[k+1] + ob2*g1*g1;
                vg[k+2] = beta2*vg[k+2] + ob2*g2*g2;
                vg[k+3] = beta2*vg[k+3] + ob2*g3*g3;
                vg[k+4] = beta2*vg[k+4] + ob2*g4*g4;
                vg[k+5] = beta2*vg[k+5] + ob2*g5*g5;
                vg[k+6] = beta2*vg[k+6] + ob2*g6*g6;
                vg[k+7] = beta2*vg[k+7] + ob2*g7*g7;

                // p -= taxaCorr * m / (sqrt(v) + eps)
                p[k] -= taxaCorr * mg[k] / (sqrtf(vg[k]) + eps);
                p[k+1] -= taxaCorr * mg[k+1] / (sqrtf(vg[k+1]) + eps);
                p[k+2] -= taxaCorr * mg[k+2] / (sqrtf(vg[k+2]) + eps);
                p[k+3] -= taxaCorr * mg[k+3] / (sqrtf(vg[k+3]) + eps);
                p[k+4] -= taxaCorr * mg[k+4] / (sqrtf(vg[k+4]) + eps);
                p[k+5] -= taxaCorr * mg[k+5] / (sqrtf(vg[k+5]) + eps);
                p[k+6] -= taxaCorr * mg[k+6] / (sqrtf(vg[k+6]) + eps);
                p[k+7] -= taxaCorr * mg[k+7] / (sqrtf(vg[k+7]) + eps);
            }
            // cauda
            float ob1 = 1.0f - beta1, ob2 = 1.0f - beta2;
            for(; k < n; k++) {
                p[k] *= pdFator;
                float gk = gd[k] * escalaClip;
                mg[k] = beta1*mg[k] + ob1*gk;
                vg[k] = beta2*vg[k] + ob2*gk*gk;
                p[k] -= taxaCorr * mg[k] / (sqrtf(vg[k]) + eps);
            }
            pos += n;
        }
    }

    void zerarGrads(Camada** camadas, int nCamadas) {
        for(int c = 0; c < nCamadas; c++) camadas[c]->zerarGrad();
    }
};