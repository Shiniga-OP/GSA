// biblis/otimis/adagrad.h
#pragma once

#include "otimizador.h"

class AdaGrad : public Otimizador {
public:
    float taxa;
    float eps;
    vector<vector<float>> somaGradPesos;
    vector<float> somaGradBias;
    
    AdaGrad(float taxa = 0.01f, float eps = 1e-8f) 
        : taxa(taxa), eps(eps) {}
    
    void att(vector<vector<float>>& pesos,
    vector<vector<float>>& gradPesos,
    vector<float>& bias,
    vector<float>& gradBias) override {
        // inicia acumuladores se estiverem vazios
        if(somaGradPesos.empty()) {
            somaGradPesos = matrizZeros(pesos.size(), pesos[0].size());
            somaGradBias = zeros(bias.size());
        }
        // atualiza pesos
        for(size_t i = 0; i < pesos.size(); i++) {
            for(size_t j = 0; j < pesos[i].size(); j++) {
                somaGradPesos[i][j] += gradPesos[i][j] * gradPesos[i][j];
                pesos[i][j] -= taxa * gradPesos[i][j] / (sqrt(somaGradPesos[i][j]) + eps);
            }
        }
        // atualiza bias
        for(size_t i = 0; i < bias.size(); i++) {
            somaGradBias[i] += gradBias[i] * gradBias[i];
            bias[i] -= taxa * gradBias[i] / (sqrt(somaGradBias[i]) + eps);
        }
    }
};