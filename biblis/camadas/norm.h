// biblis/camadas/norm.h
#pragma once
#include "camada.h"

// normalização por camada: normaliza cada vetor individualmente
// diferente de LoteNorm que normaliza por caracteristica entre amostras
// adequada para sequências de texto onde o tamanho do lote varia
// propagação:  x_norm = (x - media) / sqrt(var + eps)
// saida  = gamma * x_norm + beta
// retropropagação: gradientes de gamma, beta e da entrada

class CamadaNorm : public Camada {
public:
    size_t dim;
    float epsilon;

    vector<float> gamma;
    vector<float> beta;
    vector<float> gradGamma;
    vector<float> gradBeta;

    vector<float> entradaNormCache;
    float desvioCache;

    CamadaNorm(size_t dim, float epsilon = 1e-5f, const string& nome = "camadanorm")
        : Camada(nome), dim(dim), epsilon(epsilon), desvioCache(1.0f) {
        tipo = "CamadaNorm";
        gamma = vector<float>(dim, 1.0f);
        beta = vector<float>(dim, 0.0f);
        gradGamma = vector<float>(dim, 0.0f);
        gradBeta = vector<float>(dim, 0.0f);
    }

    vector<float> prop(const vector<float>& entrada) override {
        if(entrada.size() != dim)
            throw invalid_argument("[" + nome + "]: dimensão de entrada incorreta");

        float media = 0.0f;
        for(size_t i = 0; i < dim; i++) media += entrada[i];
        media /= (float)dim;

        float variancia = 0.0f;
        for(size_t i = 0; i < dim; i++) {
            float d = entrada[i] - media;
            variancia += d * d;
        }
        variancia /= (float)dim;

        desvioCache = sqrt(variancia + epsilon);

        entradaNormCache.resize(dim);
        vector<float> saida(dim);
        for(size_t i = 0; i < dim; i++) {
            entradaNormCache[i] = (entrada[i] - media) / desvioCache;
            saida[i] = gamma[i] * entradaNormCache[i] + beta[i];
        }
        return saida;
    }

    GradGenerico retroprop(const vector<float>& gradSaida) override {
        if(gradSaida.size() != dim)
            throw invalid_argument("[" + nome + "]: dimensão do gradiente incorreta");

        for(size_t i = 0; i < dim; i++) {
            gradGamma[i] += gradSaida[i] * entradaNormCache[i];
            gradBeta[i] += gradSaida[i];
        }

        vector<float> dNorm(dim);
        for(size_t i = 0; i < dim; i++) {
            dNorm[i] = gradSaida[i] * gamma[i];
        }
        float somaDNorm = 0.0f;
        float somaDNormXNorm = 0.0f;
        for(size_t i = 0; i < dim; i++) {
            somaDNorm += dNorm[i];
            somaDNormXNorm += dNorm[i] * entradaNormCache[i];
        }
        float mediaDNorm = somaDNorm / (float)dim;
        float mediaDNormXNorm = somaDNormXNorm / (float)dim;

        vector<float> gradEntrada(dim);
        for(size_t i = 0; i < dim; i++) {
            gradEntrada[i] = (dNorm[i] - mediaDNorm
            - entradaNormCache[i] * mediaDNormXNorm) / desvioCache;
        }
        return GradGenerico(gradEntrada);
    }

    void att(float taxaAprendizado) override {
        if(otimizador) {
            vector<vector<float>> gammaMat = {gamma};
            vector<vector<float>> gradGammaMat = {gradGamma};
            otimizador->att(gammaMat, gradGammaMat, beta, gradBeta);
            gamma = gammaMat[0];
        } else {
            for(size_t i = 0; i < dim; i++) {
                gamma[i] -= taxaAprendizado * gradGamma[i];
                beta[i] -= taxaAprendizado * gradBeta[i];
            }
        }
    }

    void zerarGradientes() override {
        fill(gradGamma.begin(), gradGamma.end(), 0.0f);
        fill(gradBeta.begin(), gradBeta.end(), 0.0f);
    }

    bool temParametros() const override { return true; }
    size_t numParametros() const override { return 2 * dim; }

    void salvar(const string& arquivo) const override {
        ofstream a(arquivo, ios::binary);
        if(!a) throw runtime_error("[" + nome + "]: falha ao abrir arquivo para salvar");
        a.write(reinterpret_cast<const char*>(&dim), sizeof(dim));
        a.write(reinterpret_cast<const char*>(&epsilon), sizeof(epsilon));
        a.write(reinterpret_cast<const char*>(gamma.data()), dim * sizeof(float));
        a.write(reinterpret_cast<const char*>(beta.data()), dim * sizeof(float));
    }

    void carregar(const string& arquivo) override {
        ifstream a(arquivo, ios::binary);
        if(!a) throw runtime_error("[" + nome + "]: falha ao abrir arquivo para carregar");
        a.read(reinterpret_cast<char*>(&dim), sizeof(dim));
        a.read(reinterpret_cast<char*>(&epsilon), sizeof(epsilon));
        gamma.resize(dim);
        beta.resize(dim);
        gradGamma.assign(dim, 0.0f);
        gradBeta.assign(dim, 0.0f);
        a.read(reinterpret_cast<char*>(gamma.data()), dim * sizeof(float));
        a.read(reinterpret_cast<char*>(beta.data()), dim * sizeof(float));
    }
};