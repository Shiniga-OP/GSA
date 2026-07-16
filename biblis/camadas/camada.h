// biblis/camadas/camada.h
#pragma once
#include <stdlib.h>
#include <string.h>
#include "../ativas.h"

struct Grad {
    float* dados;
    int tam;

    void alocar(int t) {
        tam = t;
        dados = (float*)calloc(t, sizeof(float));
    }

    void liberar() {
        free(dados);
        dados = nullptr;
        tam = 0;
    }

    void zerar() {
        memset(dados, 0, tam * sizeof(float));
    }

    void acumular(const Grad& outro) {
        for(int i = 0; i < tam; i++) {
            dados[i] += outro.dados[i];
        }
    }
};

struct Camada {
    int grupos = 2;
    float (*ativa)(float) = nullptr;
    float (*derivada)(float) = nullptr;

    void defAtivacao(const char* nome) {
        if(strcmp(nome, "sigmoid") == 0) {
            ativa = sigmoid;
            derivada = derivadaSigmoid;
        } else if(strcmp(nome, "tanh") == 0) {
            ativa = tanhF;
            derivada = derivadaTanh;
        } else if(strcmp(nome, "relu") == 0) {
            ativa = ReLU;
            derivada = derivadaReLU;
        } else if(strcmp(nome, "leakyrelu") == 0) {
            ativa = leakyReLU;
            derivada = derivadaLeakyReLU;
        } else if(strcmp(nome, "swish") == 0) {
            ativa = swish;
            derivada = derivadaSwish;
        } else if(strcmp(nome, "hardswish") == 0) {
            ativa = hardSwish;
            derivada = derivadaHardSwish;
        } else if(strcmp(nome, "gelu") == 0) {
            ativa = GELU;
            derivada = derivadaGELU;
        } else if(strcmp(nome, "elu") == 0) {
            ativa = ELU;
            derivada = derivadaELU;
        } else if(strcmp(nome, "selu") == 0) {
            ativa = SELU;
            derivada = derivadaSELU;
        } else if(strcmp(nome, "mish") == 0) {
            ativa = mish;
            derivada = derivadaMish;
        } else if(strcmp(nome, "softsign") == 0) {
            ativa = softsign;
            derivada = derivadaSoftsign;
        } else if(strcmp(nome, "softplus") == 0) {
            ativa = softplus;
            derivada = sigmoid;
        } else if(strcmp(nome, "silu") == 0) {
            ativa = SiLU;
            derivada = derivadaSwish;
        } else if(strcmp(nome, "gaussian") == 0) {
            ativa = gaussian;
            derivada = derivadaGaussian;
        } else if(strcmp(nome, "bentidentity") == 0) {
            ativa = bentIdentity;
            derivada = derivadaBentIdentity;
        }
    }
    virtual void inicializar(const char* metodo) = 0;
    virtual void prop(const float* entrada, float* saida) = 0;
    virtual void retroprop(const float* gradSaida, float* gradEntrada) = 0;
    virtual int numParams() = 0;
    virtual void params(float** saida, int* tams) = 0;
    virtual void gradParams(float** saida, int* tams) = 0;
    virtual void zerarGrad() = 0;
    virtual ~Camada() = default;
};