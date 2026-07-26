// biblis/camadas/multicabeca.h
#pragma once
#include "camada.h"
#include "../inicias.h"
#include <math.h>
#include <string.h>
#include <stdlib.h>

// dim: dimensão do modelo(divisivel por nCab)
// nCab: numero de cabeças
// seqMax: comprimento maximo de sequencia

// pesos(linha-major, sem bias):
//  Pq, Pk, Pv: [dim x dim]
//  Po: [dim x dim]

// RoPE aplicado em Q e K por cabeça antes do produto escalar
// softmax estavel(subtrai max por linha)
// retroprop() recebe/devolve gradiente achatado [seq * dim]
struct MultiCabeca : Camada {
    int dim;
    int nCab;
    int dCab; // dim / nCab
    int seqMax;

    // pesos
    float* Pq;
    float* Pk;
    float* Pv;
    float* Po;
    // gradientes de pesos
    float* gPq;
    float* gPk;
    float* gPv;
    float* gPo;

    // buffers internos(alocados em inicializar uma vez, reutilizados)
    float* ultEnt; // [seqMax * dim], copia da entrada
    float* Q; // [seqMax * dim]
    float* K; // [seqMax * dim]
    float* V; // [seqMax * dim]
    float* Qr; // [seqMax * dim], Q apos RoPE
    float* Kr; // [seqMax * dim], K apos RoPE
    float* A; // [nCab * seqMax * seqMax], pontos pre-softmax
    float* P; // [nCab * seqMax * seqMax], probs pos-softmax
    float* ctx; // [seqMax * dim], contexto pre-Po
    float* saida; // [seqMax * dim], saida final

    int seqAtual; // comprimento da sequencia na ultima prop()

    // === construtor ===
    MultiCabeca(int dim_, int nCab_, int seqMax_) {
        dim = dim_;
        nCab = nCab_;
        dCab = dim_ / nCab_;
        seqMax = seqMax_;

        int dd = dim * dim;

        Pq = (float*)malloc(dd * sizeof(float));
        Pk = (float*)malloc(dd * sizeof(float));
        Pv = (float*)malloc(dd * sizeof(float));
        Po = (float*)malloc(dd * sizeof(float));

        gPq = (float*)calloc(dd, sizeof(float));
        gPk = (float*)calloc(dd, sizeof(float));
        gPv = (float*)calloc(dd, sizeof(float));
        gPo = (float*)calloc(dd, sizeof(float));

        ultEnt = (float*)malloc(seqMax * dim * sizeof(float));
        Q = (float*)malloc(seqMax * dim * sizeof(float));
        K = (float*)malloc(seqMax * dim * sizeof(float));
        V = (float*)malloc(seqMax * dim * sizeof(float));
        Qr = (float*)malloc(seqMax * dim * sizeof(float));
        Kr = (float*)malloc(seqMax * dim * sizeof(float));
        A = (float*)malloc(nCab * seqMax * seqMax * sizeof(float));
        P = (float*)malloc(nCab * seqMax * seqMax * sizeof(float));
        ctx = (float*)malloc(seqMax * dim * sizeof(float));
        saida = (float*)malloc(seqMax * dim * sizeof(float));

        seqAtual = 0;
        grupos = 4;
        inicializar("xavier");
    }

    ~MultiCabeca() override {
        free(Pq);
        free(Pk);
        free(Pv);
        free(Po);
        free(gPq);
        free(gPk);
        free(gPv);
        free(gPo);
        free(ultEnt);
        free(Q);
        free(K);
        free(V);
        free(Qr);
        free(Kr);
        free(A);
        free(P);
        free(ctx);
        free(saida);
    }

    // === inicialização ===
    void inicializar(const char* metodo) override {
        int dd = dim * dim;
        if(strcmp(metodo, "xavier") == 0) {
            iniXavier(Pq, dd, dim, dim);
            iniXavier(Pk, dd, dim, dim);
            iniXavier(Pv, dd, dim, dim);
            iniXavier(Po, dd, dim, dim);
        } else if(strcmp(metodo, "he") == 0) {
            iniHe(Pq, dd, dim);
            iniHe(Pk, dd, dim);
            iniHe(Pv, dd, dim);
            iniHe(Po, dd, dim);
        } else if(strcmp(metodo, "zeros") == 0) {
            iniZeros(Pq, dd); iniZeros(Pk, dd);
            iniZeros(Pv, dd); iniZeros(Po, dd);
        } else {
            iniXavier(Pq, dd, dim, dim);
            iniXavier(Pk, dd, dim, dim);
            iniXavier(Pv, dd, dim, dim);
            iniXavier(Po, dd, dim, dim);
        }
    }
    //  RoPE: rotaciona pares(x[2i], x[2i+1]) pelo angulo pos/10000^(2i/dCab)
    //  opera no local sobre vetor de dimensão dCab na posição pos
    static void rope(float* v, int pos, int dCab) {
        for(int i = 0; i < dCab / 2; i++) {
            float theta = (float)pos / powf(10000.0f, 2.0f * i / (float)dCab);
            float c = cosf(theta);
            float s = sinf(theta);
            float x0 = v[2*i];
            float x1 = v[2*i + 1];
            v[2*i] = x0 * c - x1 * s;
            v[2*i + 1] = x0 * s + x1 * c;
        }
    }
    // gradiente de RoPE: dL/d(entrada) dado dL/d(saida rotacionada)
    // rotação inversa = transposta da matriz de rotação = rotação por -theta
    static void ropeGrad(const float* gSai, float* gEnt, int pos, int dCab) {
        for(int i = 0; i < dCab / 2; i++) {
            float theta = (float)pos / powf(10000.0f, 2.0f * i / (float)dCab);
            float c = cosf(theta);
            float s = sinf(theta);
            float g0 = gSai[2*i];
            float g1 = gSai[2*i + 1];
            // transposta: [c, s; -s, c]
            gEnt[2*i] = g0 * c + g1 * s;
            gEnt[2*i + 1] = -g0 * s + g1 * c;
        }
    }
    // GEMM auxiliar: C += A * B(sem bias)
    // A: [M x K], B: [K x N], C: [M x N]
    static void gemm(const float* A_, const float* B_, float* C_,
    int M, int K_, int N) {
        for(int m = 0; m < M; m++) {
            for(int k = 0; k < K_; k++) {
                float a = A_[m * K_ + k];
                if(a == 0.0f) continue;
                for(int n = 0; n < N; n++) {
                    C_[m * N + n] += a * B_[k * N + n];
                }
            }
        }
    }
    // prop(entrada, saida_saida)
    // entrada: [seq * dim](seq = seqAtual, definido externamente)
    // saida_: [seq * dim]
    // seqAtual deve ser definido antes de chamar prop()
    void prop(const float* entrada, float* saida_) override {
        int seq = seqAtual;
        float escala = 1.0f / sqrtf((float)dCab);

        memcpy(ultEnt, entrada, seq * dim * sizeof(float));

        // === projeções lineares Q = ent * Pq^T, K = ent * Pk^T, V = ent * Pv^T ===
        // Pq[dim x dim], entrada[seq x dim] → Q[seq x dim]
        memset(Q, 0, seq * dim * sizeof(float));
        memset(K, 0, seq * dim * sizeof(float));
        memset(V, 0, seq * dim * sizeof(float));
        memset(ctx, 0, seq * dim * sizeof(float));

        // Q[t, j] = soma_i entrada[t,i] * Pq[j,i](Pq linha = saida, col = entrada)
        for(int t = 0; t < seq; t++) {
            for(int j = 0; j < dim; j++) {
                float s = 0.0f;
                const float* pl = Pq + j * dim;
                const float* el = entrada + t * dim;
                for(int i = 0; i < dim; i++) s += el[i] * pl[i];
                Q[t * dim + j] = s;
            }
        }
        for(int t = 0; t < seq; t++) {
            for(int j = 0; j < dim; j++) {
                float s = 0.0f;
                const float* pl = Pk + j * dim;
                const float* el = entrada + t * dim;
                for(int i = 0; i < dim; i++) {
                    s += el[i] * pl[i];
                }
                K[t * dim + j] = s;
            }
        }
        for(int t = 0; t < seq; t++) {
            for(int j = 0; j < dim; j++) {
                float s = 0.0f;
                const float* pl = Pv + j * dim;
                const float* el = entrada + t * dim;
                for(int i = 0; i < dim; i++) {
                    s += el[i] * pl[i];
                }
                V[t * dim + j] = s;
            }
        }

        // === RoPE em Q e K por cabeça ===
        memcpy(Qr, Q, seq * dim * sizeof(float));
        memcpy(Kr, K, seq * dim * sizeof(float));
        for(int c = 0; c < nCab; c++) {
            for(int t = 0; t < seq; t++) {
                rope(Qr + t * dim + c * dCab, t, dCab);
                rope(Kr + t * dim + c * dCab, t, dCab);
            }
        }
        // === pontos, softmax e contexto por cabeça ===
        for(int c = 0; c < nCab; c++) {
            float* Ac = A + c * seq * seq;
            float* Pc = P + c * seq * seq;

            // A[q,k] = escala * dot(Qr[q,c*dCab..], Kr[k,c*dCab..])
            // mascara causal: q so pode atender a k <= q (nunca o futuro)
            for(int q = 0; q < seq; q++) {
                const float* qv = Qr + q * dim + c * dCab;
                float mx = -1e30f;
                for(int k = 0; k <= q; k++) {
                    const float* kv = Kr + k * dim + c * dCab;
                    float dot = 0.0f;
                    for(int d = 0; d < dCab; d++) dot += qv[d] * kv[d];
                    Ac[q * seq + k] = dot * escala;
                    if(Ac[q * seq + k] > mx) mx = Ac[q * seq + k];
                }
                // softmax estavel, somente sobre k <= q
                float soma = 0.0f;
                for(int k = 0; k <= q; k++) {
                    Pc[q * seq + k] = expf(Ac[q * seq + k] - mx);
                    soma += Pc[q * seq + k];
                }
                float inv = 1.0f / soma;
                for(int k = 0; k <= q; k++) Pc[q * seq + k] *= inv;
                for(int k = q + 1; k < seq; k++) Pc[q * seq + k] = 0.0f; // futuro: peso zero
            }
            // ctx[q, c*dCab..] = soma_{k<=q} P[q,k] * V[k, h*dCab..]
            for(int q = 0; q < seq; q++) {
                float* cv = ctx + q * dim + c * dCab;
                const float* pv = Pc + q * seq;
                for(int k = 0; k <= q; k++) {
                    const float* vv = V + k * dim + c * dCab;
                    float p = pv[k];
                    for(int d = 0; d < dCab; d++) {
                        cv[d] += p * vv[d];
                    }
                }
            }
        }
        // === projeção de saida: saida = ctx * Po^T ===
        memset(saida_, 0, seq * dim * sizeof(float));
        for(int t = 0; t < seq; t++) {
            for(int j = 0; j < dim; j++) {
                float s = 0.0f;
                const float* pl = Po + j * dim;
                const float* cl = ctx + t * dim;
                for(int i = 0; i < dim; i++) {
                    s += cl[i] * pl[i];
                }
                saida_[t * dim + j] = s;
            }
        }
        memcpy(saida, saida_, seq * dim * sizeof(float));
    }

    // retroprop(gradSaida, gradEntrada)
    // gradSaida: dL/d(saida), [seq * dim]
    // gradEntrada: dL/d(entrada), [seq * dim](pode ser nullptr)
    // acumula em gPq, gPk, gPv, gPo
    void retroprop(const float* gradSaida, float* gradEntrada) override {
        int seq = seqAtual;
        float escala = 1.0f / sqrtf((float)dCab);

        // buffers temporários no heap (seq pode ser grande)
        float* gCtx = (float*)calloc(seq * dim, sizeof(float));
        float* gQr = (float*)calloc(seq * dim, sizeof(float));
        float* gKr = (float*)calloc(seq * dim, sizeof(float));
        float* gV = (float*)calloc(seq * dim, sizeof(float));
        float* gQ = (float*)calloc(seq * dim, sizeof(float));
        float* gK = (float*)calloc(seq * dim, sizeof(float));

        // === grad de Po e gCtx ===
        // saida[t,j] = soma_i ctx[t,i] * Po[j,i]
        // gPo[j,i] += soma_t gradSaida[t,j] * ctx[t,i]
        // gCtx[t,i] += soma_j gradSaida[t,j] * Po[j,i]
        for(int t = 0; t < seq; t++) {
            for(int j = 0; j < dim; j++) {
                float gs = gradSaida[t * dim + j];
                float* pl = Po + j * dim;
                float* gpl = gPo + j * dim;
                float* gcl = gCtx + t * dim;
                for(int i = 0; i < dim; i++) {
                    gpl[i] += gs * ctx[t * dim + i];
                    gcl[i] += gs * pl[i];
                }
            }
        }
        // === grad por cabeça: gCtx -> gQr, gKr, gV, gPv ===
        for(int c = 0; c < nCab; c++) {
            const float* Pc = P + c * seq * seq;
            float* gAc = (float*)calloc(seq * seq, sizeof(float));

            // ctx[q, c*dCab..] = soma_k P[q,k] * V[k, c*dCab..]
            // gV[k, c*dCab..] += soma_q P[q,k] * gCtx[q, c*dCab..]
            // gP[q,k] = soma_d gCtx[q, c*dCab+d] * V[k, c*dCab+d]
            for(int q = 0; q < seq; q++) {
                const float* gcv = gCtx + q * dim + c * dCab;
                const float* pv = Pc + q * seq;
                for(int k = 0; k < seq; k++) {
                    float* gvv = gV + k * dim + c * dCab;
                    const float* vv = V + k * dim + c * dCab;
                    float dot = 0.0f;
                    for(int d = 0; d < dCab; d++) {
                        gvv[d] += pv[k] * gcv[d];
                        dot += gcv[d] * vv[d];
                    }
                    gAc[q * seq + k] = dot; // gP[q,k](antes do softmax grad)
                }
            }
            // grad do softmax: gA[q,k] = P[q,k] * (gP[q,k] - soma_j P[q,j]*gP[q,j])
            for(int q = 0; q < seq; q++) {
                const float* pv = Pc + q * seq;
                float* gav = gAc + q * seq;
                float dot = 0.0f;
                for(int k = 0; k < seq; k++) dot += pv[k] * gav[k];
                for(int k = 0; k < seq; k++) {
                    gav[k] = pv[k] * (gav[k] - dot);
                }
            }
            // grad de escala: gPonto = gA * escala
            // A[q,k] = escala * dot(Qr[q], Kr[k])
            // gQr[q, c*dCab..] += soma_k gA[q,k]*escala * Kr[k, c*dCab..]
            // gKr[k, c*dCab..] += soma_q gA[q,k]*escala * Qr[q, c*dCab..]
            for(int q = 0; q < seq; q++) {
                float* gqv = gQr + q * dim + c * dCab;
                for(int k = 0; k < seq; k++) {
                    float gs = gAc[q * seq + k] * escala;
                    const float* kv  = Kr + k * dim + c * dCab;
                    const float* qv  = Qr + q * dim + c * dCab;
                    float* gkv = gKr + k * dim + c * dCab;
                    for(int d = 0; d < dCab; d++) {
                        gqv[d] += gs * kv[d];
                        gkv[d] += gs * qv[d];
                    }
                }
            }
            free(gAc);
        }
        // === grad de RoPE: gQr -> gQ, gKr -> gK ===
        for(int c = 0; c < nCab; c++) {
            for(int t = 0; t < seq; t++) {
                ropeGrad(gQr + t * dim + c * dCab, gQ + t * dim + c * dCab, t, dCab);
                ropeGrad(gKr + t * dim + c * dCab, gK + t * dim + c * dCab, t, dCab);
            }
        }
        // === grad de Pq, Pk, Pv e gradEntrada ===
        // Q[t,j] = soma_i ent[t,i] * Pq[j,i]
        // gPq[j,i] += soma_t gQ[t,j] * ent[t,i]
        // gEnt[t,i] += soma_j gQ[t,j] * Pq[j,i](idem Pk, Pv)
        if(gradEntrada) memset(gradEntrada, 0, seq * dim * sizeof(float));

        for(int t = 0; t < seq; t++) {
            const float* el = ultEnt + t * dim;
            for(int j = 0; j < dim; j++) {
                float gq = gQ[t * dim + j];
                float gk = gK[t * dim + j];
                float gv = gV[t * dim + j];
                float* gPql = gPq + j * dim;
                float* gPkl = gPk + j * dim;
                float* gPvl = gPv + j * dim;
                for(int i = 0; i < dim; i++) {
                    gPql[i] += gq * el[i];
                    gPkl[i] += gk * el[i];
                    gPvl[i] += gv * el[i];
                }
                if(gradEntrada) {
                    float* gel = gradEntrada + t * dim;
                    const float* Pql = Pq + j * dim;
                    const float* Pkl = Pk + j * dim;
                    const float* Pvl = Pv + j * dim;
                    for(int i = 0; i < dim; i++) {
                        gel[i] += gq * Pql[i] + gk * Pkl[i] + gv * Pvl[i];
                    }
                }
            }
        }
        free(gCtx);
        free(gQr);
        free(gKr);
        free(gV);
        free(gQ);
        free(gK);
    }

    // === interface Camada ===
    int numParams() override { return 4 * dim * dim; }

    void params(float** saida_, int* tams) override {
        saida_[0] = Pq;
        tams[0] = dim * dim;
        saida_[1] = Pk;
        tams[1] = dim * dim;
        saida_[2] = Pv;
        tams[2] = dim * dim;
        saida_[3] = Po;
        tams[3] = dim * dim;
    }

    void gradParams(float** saida_, int* tams) override {
        saida_[0] = gPq;
        tams[0] = dim * dim;
        saida_[1] = gPk;
        tams[1] = dim * dim;
        saida_[2] = gPv;
        tams[2] = dim * dim;
        saida_[3] = gPo;
        tams[3] = dim * dim;
    }

    void zerarGrad() override {
        int dd = dim * dim;
        memset(gPq, 0, dd * sizeof(float));
        memset(gPk, 0, dd * sizeof(float));
        memset(gPv, 0, dd * sizeof(float));
        memset(gPo, 0, dd * sizeof(float));
    }
};