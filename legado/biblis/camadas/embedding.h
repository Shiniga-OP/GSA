// biblis/camadas/embedding.h
#pragma once
#include "camada.h"

// lookup treinável: id inteiro -> vetor float de dimensão dim
// a tabela E tem shape [vocabTam x dim]
// prop: retorna E[id]
// retroprop: acumula gradiente só na linha E[id] que foi usada

class Embedding : public Camada {
public:
    size_t vocabTam;
    size_t dim;

    vector<vector<float>> E; // [vocabTam x dim]
    vector<vector<float>> gradE; // gradientes acumulados

    // cache para retropropagação
    size_t idCache;

    Embedding(size_t vocabTam, size_t dim, const string& nome = "embedding")
        : Camada(nome), vocabTam(vocabTam), dim(dim), idCache(0) {
        tipo = "Embedding";
        // Xavier sobre fan_in = vocabTam, fan_saida = dim
        E = iniPesosXavier((int)vocabTam, (int)dim);
        gradE = matrizZeros(vocabTam, dim);
    }

    // prop por id inteiro interface principal
    vector<float> prop(size_t id) {
        if(id >= vocabTam)
            throw invalid_argument("[" + nome + "]: id " + to_string(id) + " fora do vocabulário");
        idCache = id;
        return E[id];
    }

    // prop(vetor) não faz sentido aqui; lança erro claro
    vector<float> prop(const vector<float>& entrada) override {
        if(entrada.size() != 1)
            throw runtime_error("[" + nome + "]: use prop(size_t id) para embedding");
        return prop((size_t)entrada[0]);
    }

    // retroprop: gradSaida(dim,) -> acumula em gradE[idCache]
    // retorna gradiente zero em relação a entrada(id é discreto)
    GradGenerico retroprop(const vector<float>& gradSaida) override {
        if(gradSaida.size() != dim)
            throw invalid_argument("[" + nome + "]: dimensão do gradiente incorreta");
        for(size_t j = 0; j < dim; j++) {
            gradE[idCache][j] += gradSaida[j];
        }
        // entrada é um indice discreto não ha gradiente real
        return GradGenerico(vector<float>(1, 0.0f));
    }

    void att(float taxaAprendizado) override {
        if(otimizador) {
            vector<float> biasZero(1, 0.0f);
            vector<float> gradBiasZero(1, 0.0f);
            otimizador->att(E, gradE, biasZero, gradBiasZero);
        } else {
            for(size_t i = 0; i < vocabTam; i++) {
                for(size_t j = 0; j < dim; j++) {
                    E[i][j] -= taxaAprendizado * gradE[i][j];
                }
            }
        }
    }

    void zerarGradientes() override {
        for(auto& linha : gradE) fill(linha.begin(), linha.end(), 0.0f);
    }

    bool temParametros() const override { return true; }
    size_t numParametros() const override { return vocabTam * dim; }

    void salvar(const string& arquivo) const override {
        ofstream a(arquivo, ios::binary);
        if(!a) throw runtime_error("[" + nome + "]: falha ao abrir arquivo para salvar");
        a.write(reinterpret_cast<const char*>(&vocabTam), sizeof(vocabTam));
        a.write(reinterpret_cast<const char*>(&dim), sizeof(dim));
        for(const auto& linha : E) {
            a.write(reinterpret_cast<const char*>(linha.data()), dim * sizeof(float));
        }
    }

    void carregar(const string& arquivo) override {
        ifstream a(arquivo, ios::binary);
        if(!a) throw runtime_error("[" + nome + "]: falha ao abrir arquivo para carregar");
        a.read(reinterpret_cast<char*>(&vocabTam), sizeof(vocabTam));
        a.read(reinterpret_cast<char*>(&dim), sizeof(dim));
        E.assign(vocabTam, vector<float>(dim));
        for(auto& linha : E) {
            a.read(reinterpret_cast<char*>(linha.data()), dim * sizeof(float));
        }
        gradE = matrizZeros(vocabTam, dim);
    }
};