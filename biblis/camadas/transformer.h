// biblis/camadas/transformer.h
// Bloco Transformer decoder(Pre-LN, causal).

// arquitetura por bloco:
//   x' = x + MHA(LN1(x)) <- atenção com resíduo
//   x''= x'+ FFN(LN2(x')) <- FFN com resíduo

// FFN:
//   h = GELU(LN2(x') * P1 + b1) dim -> dimFF
//   saida = h * P2 + b2 dimFF -> dim

// buffers de ativação:
//   todos dimensionados para seqMax, nenhum sobrescrito entre passos.
//   defSeq(seq) grava seqAtual e propaga para MHA(via campo seqAtual);
//   Norm e Densa operam por token e são chamadas em loop dentro de prop/retroprop.

// gradiente:
//   retroprop recebe gradSaida[seq*dim], devolve gradEntrada [seq*dim].
//   acumula gradientes em todas as sub-camadas.
//   a conexão residual é somada diretamente(gradiente = 1).

// grupos = MHA(4) + LN1(2) + Densa1(2) + LN2(2) + Densa2(2) = 12
#pragma once
#include "camada.h"
#include "multicabeca.h"
#include "norm.h"
#include "densa.h"
#include <string.h>
#include <stdlib.h>

struct BlocoTransformer : Camada {
    int dim;
    int dimFF;
    int seqMax;
    int seqAtual;

    MultiCabeca* mha; // atenção multi-cabeça
    Norm* ln1; // camada norm antes da atenção
    Norm* ln2; // camada norm antes da FFN
    Densa* ff1; // projeção FFN: dim -> dimFF (GELU)
    Densa* ff2; // projeção FFN: dimFF -> dim (linear)

    // buffers intermediarios[seqMax * dim] (ou[seqMax * dimFF])
    // ln1/ln2 preenchem estes buffers token a token(Norm opera sobre[dim] único)
    float* bufLN1; // saida de LN1
    float* bufMHA; // saida de MHA
    float* bufRes1; // x + bufMHA(entrada de LN2)
    float* bufLN2; // saida de LN2
    float* bufFF1; // saida de ff1(pré-ativação já aplicada dentro de Densa)
    // nota: a ativação GELU já é aplicada dentro de ff1 (Densa com "gelu")
    // bufFF1 guarda a saída de ff1 — usada como entrada de ff2

    // buffers de gradiente temporários para retroprop
    float* gBufFF2; // [seqMax * dimFF]
    float* gBufFF1; // [seqMax * dim]
    float* gBufLN2; // [seqMax * dim]
    float* gBufMHA; // [seqMax * dim]
    float* gBufLN1; // [seqMax * dim]

    // construtor
    // dim_: dimensão do modelo
    // nCab_: numero de cabeças da atenção
    // dimFF_: dimensão interna da FFN (tipicamente 4*dim)
    // seqMax_: comprimento maximo de sequencia

    BlocoTransformer(int dim_, int nCab_, int dimFF_, int seqMax_) {
        dim = dim_;
        dimFF = dimFF_;
        seqMax = seqMax_;
        seqAtual = 1;

        mha = new MultiCabeca(dim, nCab_, seqMax);
        ln1 = new Norm(dim, 1e-5f, seqMax);
        ln2 = new Norm(dim, 1e-5f, seqMax);
        ff1 = new Densa(dim, dimFF, "gelu");
        ff2 = new Densa(dimFF, dim, ""); // sem ativação(linear)

        grupos = mha->grupos + ln1->grupos + ln2->grupos + ff1->grupos + ff2->grupos;

        bufLN1 = (float*)malloc(seqMax * dim   * sizeof(float));
        bufMHA = (float*)malloc(seqMax * dim   * sizeof(float));
        bufRes1 = (float*)malloc(seqMax * dim   * sizeof(float));
        bufLN2 = (float*)malloc(seqMax * dim   * sizeof(float));
        bufFF1 = (float*)malloc(seqMax * dimFF * sizeof(float));

        gBufFF2  = (float*)malloc(seqMax * dimFF * sizeof(float));
        gBufFF1 = (float*)malloc(seqMax * dim   * sizeof(float));
        gBufLN2 = (float*)malloc(seqMax * dim   * sizeof(float));
        gBufMHA = (float*)malloc(seqMax * dim   * sizeof(float));
        gBufLN1 = (float*)malloc(seqMax * dim   * sizeof(float));
    }

    ~BlocoTransformer() override {
        delete mha; delete ln1; delete ln2; delete ff1; delete ff2;
        free(bufLN1);
        free(bufMHA);
        free(bufRes1);
        free(bufLN2);
        free(bufFF1);
        free(gBufFF2);
        free(gBufFF1);
        free(gBufLN2);
        free(gBufMHA);
        free(gBufLN1);
    }
    // não é virtual em Camada(so MultiCabeca usa noção de sequência inteira,
    // via campo publico seqAtual; Norm e Densa operam por token, sem estado de seq)
    void defSeq(int seq) {
        seqAtual = seq;
        mha->seqAtual = seq;
        // Norm e Densa operam por token: chamadas em loop dentro de prop()/retroprop()
    }

    void inicializar(const char* metodo) override {
        mha->inicializar(metodo);
        ln1->inicializar(metodo);
        ln2->inicializar(metodo);
        ff1->inicializar(metodo);
        ff2->inicializar(metodo);
    }
    // prop
    // entrada: [seqAtual * dim]
    // saida: [seqAtual * dim]

    void prop(const float* entrada, float* saida) override {
        int seq = seqAtual;
        int dimSq  = seq * dim;

        // --- bloco de atenção(Pre-LN) ---
        // LN1(x) -> bufLN1, por token(Norm opera sobre um vetor [dim] de cada vez)
        // defPos(0) reinicia o contador de estado por token antes do ciclo
        ln1->defPos(0);
        for(int t = 0; t < seq; t++) {
            ln1->prop(entrada + t*dim, bufLN1 + t*dim);
        }
        // MHA(bufLN1) -> bufMHA (MultiCabeca processa a sequência inteira; seqAtual já setado via defSeq)
        mha->prop(bufLN1, bufMHA);

        // residuo: bufRes1 = x + MHA(LN1(x))
        for(int i = 0; i < dimSq; i++) { bufRes1[i] = entrada[i] + bufMHA[i];
        }
        // --- bloco FFN(Pre-LN) ---
        // LN2(bufRes1) -> bufLN2, por token
        ln2->defPos(0);
        for(int t = 0; t < seq; t++) {
            ln2->prop(bufRes1 + t*dim, bufLN2 + t*dim);
        }
        // ff1: dim -> dimFF, por token, com GELU interno
        for(int t = 0; t < seq; t++) {
            ff1->prop(bufLN2 + t*dim, bufFF1 + t*dimFF);
        }
        // ff2: dimFF -> dim, por token, linear
        // saída vai direto para o buffer de saída e depois somamos o resíduo
        for(int t = 0; t < seq; t++) {
            float* st = saida + t*dim;
            ff2->prop(bufFF1 + t*dimFF, st);
            // residuo: saida[t] = ff2(ff1(LN2(bufRes1[t]))) + bufRes1[t]
            const float* r = bufRes1 + t*dim;
            for(int i = 0; i < dim; i++) st[i] += r[i];
        }
    }

    // retroprop
    // gradSaida: [seqAtual * dim]
    // gradEntrada: [seqAtual * dim](pode ser nullptr)

    void retroprop(const float* gradSaida, float* gradEntrada) override {
        int seq = seqAtual;
        int dimSq = seq * dim;

        // caminho de volta do bloco FFN
        // saida[t] = ff2(ff1(LN2(bufRes1[t]))) + bufRes1[t]
        // grad da saida split: vai para ff2 e para bufRes1 (resíduo)

        // grad atraves de ff2 por token -> gBufFF2
        for(int t = 0; t < seq; t++) {
            // ff2->retroprop acumula em gradP/gradB de ff2,
            // devolve grad da entrada (bufFF1[t]) em gBufFF2[t]
            ff2->retroprop(gradSaida + t*dim, gBufFF2 + t*dimFF);
        }
        // grad atraves de ff1 por token -> gBufFF1 (grad de bufLN2[t])
        for(int t = 0; t < seq; t++) {
            ff1->retroprop(gBufFF2 + t*dimFF, gBufFF1 + t*dim);
        }
        // grad atraves de LN2, por token (Norm opera sobre [dim] único) -> gBufLN2
        // defPos(0) garante que o estado lido em cada retroprop() corresponda
        // ao token t certo (mesma ordem 0..seq-1 usada no prop)
        ln2->defPos(0);
        for(int t = 0; t < seq; t++) {
            ln2->retroprop(gBufFF1 + t*dim, gBufLN2 + t*dim);
        }
        // grad total para bufRes1:
        //   do resíduo da saída: gradSaida diretamente (gradiente da soma = 1)
        //   de LN2: gBufLN2
        // gBufMHA reutilizado aqui temporariamente como "grad para bufRes1"
        for(int i = 0; i < dimSq; i++) { gBufMHA[i] = gradSaida[i] + gBufLN2[i];
        }
        // caminho de volta do bloco de atenção
        // bufRes1 = entrada + MHA(LN1(entrada))
        // grad de bufRes1 -> split para MHA e para entrada (resíduo)

        // grad de MHA: gBufMHA -> gBufLN1 (grad de bufLN1)
        mha->retroprop(gBufMHA, gBufLN1);

        // grad atraves de LN1, por token(Norm opera sobre[dim] unico) -> gBufFF1 reutilizado
        ln1->defPos(0);
        for(int t = 0; t < seq; t++) {
            ln1->retroprop(gBufLN1 + t*dim, gBufFF1 + t*dim);
        }
        // grad total para entrada:
        // do residuo do bloco de atenção: gBufMHA(gradiente da soma = 1)
        //   de LN1: gBufFF1
        if(gradEntrada) {
            for(int i = 0; i < dimSq; i++) {
                gradEntrada[i] = gBufMHA[i] + gBufFF1[i];
            }
        }
    }
    // interface Camada
    // grupos(membro, setado no construtor) = MHA(4)+LN1(2)+LN2(2)+ff1(2)+ff2(2) = 12
    int numParams() override {
        return mha->numParams() + ln1->numParams() + ln2->numParams() + ff1->numParams() + ff2->numParams();
    }

    void params(float** saida, int* tams) override {
        int pos = 0;
        mha->params(saida + pos, tams + pos); pos += mha->grupos;
        ln1->params(saida + pos, tams + pos); pos += ln1->grupos;
        ln2->params(saida + pos, tams + pos); pos += ln2->grupos;
        ff1->params(saida + pos, tams + pos); pos += ff1->grupos;
        ff2->params(saida + pos, tams + pos);
    }

    void gradParams(float** saida, int* tams) override {
        int off = 0;
        mha->gradParams(saida + off, tams + off); off += mha->grupos;
        ln1->gradParams(saida + off, tams + off); off += ln1->grupos;
        ln2->gradParams(saida + off, tams + off); off += ln2->grupos;
        ff1->gradParams(saida + off, tams + off); off += ff1->grupos;
        ff2->gradParams(saida + off, tams + off);
    }

    void zerarGrad() override {
        mha->zerarGrad();
        ln1->zerarGrad();
        ln2->zerarGrad();
        ff1->zerarGrad();
        ff2->zerarGrad();
    }
};