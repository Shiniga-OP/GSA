// biblis/otimis/nesterov.h
#pragma once

#include "otimizador.h"

class Nesterov : public Otimizador {
public:
    float taxa;
    float momentum;
    vector<vector<float>> velocidadePesos;
    vector<float> velocidadeBias;
    
    Nesterov(float taxa = 0.01f, float momentum = 0.9f)
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
        // salva velocidade anterior
        auto velocidadePesosAntiga = velocidadePesos;
        auto velocidadeBiasAntiga = velocidadeBias;
        
        // atualiza velocidade com gradiente atual
        for(size_t i = 0; i < pesos.size(); i++) {
            for(size_t j = 0; j < pesos[i].size(); j++) {
                velocidadePesos[i][j] = momentum * velocidadePesos[i][j] - taxa * gradPesos[i][j];
            }
        }
        for(size_t i = 0; i < bias.size(); i++) {
            velocidadeBias[i] = momentum * velocidadeBias[i] - taxa * gradBias[i];
        }
        // atualiza com lookahead(nesterov)
        for(size_t i = 0; i < pesos.size(); i++) {
            for(size_t j = 0; j < pesos[i].size(); j++) {
                pesos[i][j] += -momentum * velocidadePesosAntiga[i][j] + 
                (1.0f + momentum) * velocidadePesos[i][j];
            }
        }
        for(size_t i = 0; i < bias.size(); i++) {
            bias[i] += -momentum * velocidadeBiasAntiga[i] + 
            (1.0f + momentum) * velocidadeBias[i];
        }
    }
};
