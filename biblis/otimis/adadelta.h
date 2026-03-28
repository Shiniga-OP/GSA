// biblis/otimis/adadelta.h
#pragma once

#include "otimizador.h"

class AdaDelta : public Otimizador {
public:
    float rho;
    float eps;
    vector<vector<float>> acumGradPesos;
    vector<vector<float>> acumDeltaPesos;
    vector<float> acumGradBias;
    vector<float> acumDeltaBias;
    
    AdaDelta(float rho = 0.95f, float eps = 1e-6f)
        : rho(rho), eps(eps) {}
    
    void att(vector<vector<float>>& pesos,
    vector<vector<float>>& gradPesos,
    vector<float>& bias,
    vector<float>& gradBias) override {
        // inicia acumuladores se estiverem vazios
        if(acumGradPesos.empty()) {
            acumGradPesos = matrizZeros(pesos.size(), pesos[0].size());
            acumDeltaPesos = matrizZeros(pesos.size(), pesos[0].size());
            acumGradBias = zeros(bias.size());
            acumDeltaBias = zeros(bias.size());
        }
        // atualiza pesos
        for(size_t i = 0; i < pesos.size(); i++) {
            for(size_t j = 0; j < pesos[i].size(); j++) {
                acumGradPesos[i][j] = rho * acumGradPesos[i][j] + 
                (1.0f - rho) * gradPesos[i][j] * gradPesos[i][j];
                
                float delta = sqrt(acumDeltaPesos[i][j] + eps) / 
                             sqrt(acumGradPesos[i][j] + eps) * gradPesos[i][j];
                
                pesos[i][j] -= delta;
                
                acumDeltaPesos[i][j] = rho * acumDeltaPesos[i][j] + 
                                       (1.0f - rho) * delta * delta;
            }
        }
        // atualiza bias
        for(size_t i = 0; i < bias.size(); i++) {
            acumGradBias[i] = rho * acumGradBias[i] +
            (1.0f - rho) * gradBias[i] * gradBias[i];
            
            float delta = sqrt(acumDeltaBias[i] + eps) /
            sqrt(acumGradBias[i] + eps) * gradBias[i];
            
            bias[i] -= delta;
            
            acumDeltaBias[i] = rho * acumDeltaBias[i] + (1.0f - rho) * delta * delta;
        }
    }
};