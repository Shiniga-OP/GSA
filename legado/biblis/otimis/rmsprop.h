// biblis/otimis/rmsprop.h
#pragma once

#include "otimizador.h"

class RMSprop : public Otimizador {
public:
    float taxa;
    float decaimento;
    float eps;
    vector<vector<float>> somaPesos;
    vector<float> somaBias;
    
    RMSprop(float taxa = 0.001f, float decaimento = 0.9f, float eps = 1e-8f)
        : taxa(taxa), decaimento(decaimento), eps(eps) {}
    
    void att(vector<vector<float>>& pesos,
    vector<vector<float>>& gradPesos,
    vector<float>& bias,
    vector<float>& gradBias) override {
        // inicia acumuladores se estiverem vazios
        if(somaPesos.empty()) {
            somaPesos = matrizZeros(pesos.size(), pesos[0].size());
            somaBias = zeros(bias.size());
        }
        // atualiza pesos
        for(size_t i = 0; i < pesos.size(); i++) {
            for(size_t j = 0; j < pesos[i].size(); j++) {
                somaPesos[i][j] = decaimento * somaPesos[i][j] + 
                                  (1.0f - decaimento) * gradPesos[i][j] * gradPesos[i][j];
                pesos[i][j] -= taxa * gradPesos[i][j] / (sqrt(somaPesos[i][j]) + eps);
            }
        }
        // atualiza bias
        for(size_t i = 0; i < bias.size(); i++) {
            somaBias[i] = decaimento * somaBias[i] + 
                         (1.0f - decaimento) * gradBias[i] * gradBias[i];
            bias[i] -= taxa * gradBias[i] / (sqrt(somaBias[i]) + eps);
        }
    }
};