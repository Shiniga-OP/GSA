// biblis/camadas/embedding.h
#pragma once
#include "camada.h"
#include "../inicias.h"
#include <string.h>
#include <math.h>

// embedding treinavel: tabela [vocab x dim], pesquisa por indice inteiro

// convenção de interface:
//   prop(entrada, saida)
//     entrada: int* reinterpretado como float*: sequencia de tamSeq ids
//     saida: float[tamSeq * dim], vetores correspondentes(copia)

//   retroprop(gradSaida, gradEntrada)
//     gradSaida: float[tamSeq * dim], gradiente vindo de cima
//     gradEntrada: ignorado(ids não tem gradiente)
//     acumula em gradTab via dispersão-adição

// tamSeq deve ser definido antes de cada prop()/retroprop() via defSeq()
// os ids da última prop() são gravados em ultIds[]

struct Embedding : Camada {
    int vocab; // tamanho do vocabulário
    int dim; // dimensão do embedding
    int tamSeq; // comprimento da sequência atual

    float* tabela; // [vocab * dim]
    float* gradTab; // [vocab * dim], acumulado
    int* ultIds; // [tamSeqMax]: ids da ultima prop()
    int tamSeqMax;

    // === ciclo de vida ===
    Embedding(int vocab, int dim, int tamSeqMax = 512) {
        this->vocab = vocab;
        this->dim = dim;
        this->tamSeq = 0;
        this->tamSeqMax = tamSeqMax;
        tabela = (float*)malloc(vocab * dim * sizeof(float));
        gradTab = (float*)calloc(vocab * dim, sizeof(float));
        ultIds = (int*)malloc(tamSeqMax * sizeof(int));
        inicializar("normal");
    }

    ~Embedding() override {
        free(tabela);
        free(gradTab);
        free(ultIds);
    }

    // === inicialização ===
    // "normal": N(0, 1/sqrt(dim))
    // "xavier": iniXavier sobre tabela inteira
    // "he": iniHe
    // "zeros": zeros
    void inicializar(const char* metodo) override {
        if(strcmp(metodo, "normal") == 0) {
            float escala = 1.0f / sqrtf((float)dim);
            iniNormal(tabela, vocab * dim, 0.0f, escala);
        } else if(strcmp(metodo, "xavier") == 0) {
            iniXavier(tabela, vocab * dim, dim, dim);
        } else if(strcmp(metodo, "he") == 0) {
            iniHe(tabela, vocab * dim, dim);
        } else if(strcmp(metodo, "zeros") == 0) {
            iniZeros(tabela, vocab * dim);
        } else {
            float escala = 1.0f / sqrtf((float)dim);
            iniNormal(tabela, vocab * dim, 0.0f, escala);
        }
    }

    // === propagação direta ===
    // entrada: int* reinterpretado como const float*
    // saida: float[tamSeq * dim]
    void prop(const float* entrada, float* saida) override {
        const int* ids = (const int*)entrada;
        for(int t = 0; t < tamSeq; t++) {
            int id = ids[t];
            ultIds[t] = id;
            const float* linha = tabela + id * dim;
            float* dest = saida + t * dim;
            // copia com desenrolamento x4
            int k = 0;
            for(; k <= dim - 4; k += 4) {
                dest[k] = linha[k];
                dest[k+1] = linha[k+1];
                dest[k+2] = linha[k+2];
                dest[k+3] = linha[k+3];
            }
            for(; k < dim; k++) dest[k] = linha[k];
        }
    }

    // === retropropagação ===
    // gradSaida: float[tamSeq * dim]
    // gradEntrada: ignorado
    // adição de dispersão: gradTab[id] += gradSaida[t]
    void retroprop(const float* gradSaida, float* gradEntrada) override {
        (void)gradEntrada;
        for(int t = 0; t < tamSeq; t++) {
            int id = ultIds[t];
            float* gLinha = gradTab + id * dim;
            const float* gs = gradSaida + t * dim;
            int k = 0;
            for(; k <= dim - 4; k += 4) {
                gLinha[k] += gs[k];
                gLinha[k+1] += gs[k+1];
                gLinha[k+2] += gs[k+2];
                gLinha[k+3] += gs[k+3];
            }
            for(; k < dim; k++) gLinha[k] += gs[k];
        }
    }

    // === interface Camada ===
    int numParams() override { return vocab * dim; }

    void params(float** saida, int* tams) override {
        saida[0] = tabela;
        tams[0] = vocab * dim;
    }

    void gradParams(float** saida, int* tams) override {
        saida[0] = gradTab;
        tams[0] = vocab * dim;
    }

    void zerarGrad() override {
        memset(gradTab, 0, vocab * dim * sizeof(float));
    }
};