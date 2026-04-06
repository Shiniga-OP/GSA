// biblis/otimis/sgd.h
#pragma once

#include "otimizador.h"

class SGD : public Otimizador {
public:
    float taxa;
    float momentum;
    vector<vector<float>> velocidadePesos;
    vector<float> velocidadeBias;
    
    SGD(float taxa = 0.01f, float momentum = 0.0f) 
        : taxa(taxa), momentum(momentum) {}
    
    void att(vector<vector<float>>& pesos,
    vector<vector<float>>& gradPesos,
    vector<float>& bias,
    vector<float>& gradBias) override {
        // inicia velocidades se estiverem vazias
        if(velocidadePesos.empty()) {
            velocidadePesos = matrizZeros(pesos.size(), pesos[0].size());
            velocidadeBias = zeros(bias.size());
        }
        // atualiza pesos com momentum
        for(size_t i = 0; i < pesos.size(); i++) {
            for(size_t j = 0; j < pesos[i].size(); j++) {
                velocidadePesos[i][j] = momentum * velocidadePesos[i][j] - taxa * gradPesos[i][j];
                pesos[i][j] += velocidadePesos[i][j];
            }
        }
        // atualiza bias com momentum
        for(size_t i = 0; i < bias.size(); i++) {
            velocidadeBias[i] = momentum * velocidadeBias[i] - taxa * gradBias[i];
            bias[i] += velocidadeBias[i];
        }
    }
};