// biblis/otimis/adamw.h
#pragma once

#include "otimizador.h"

class AdamW : public Otimizador {
public:
    float taxa, b1, b2, eps;
    float pesoDecaimento;
    int t = 0;
    vector<vector<float>> m_pesos, v_pesos;
    vector<float> m_bias, v_bias;
    
    AdamW(float taxa = 0.001f, float pesoDecaimento = 0.01f)
        : taxa(taxa), b1(0.9f), b2(0.999f), eps(1e-8f), pesoDecaimento(pesoDecaimento) {}
    
    void att(vector<vector<float>>& pesos,
    vector<vector<float>>& gradPesos,
    vector<float>& bias,
    vector<float>& gradBias) override {
        t++;
        
        // inicia cache se vazio
        if(m_pesos.empty()) {
            m_pesos = matrizZeros(pesos.size(), pesos[0].size());
            v_pesos = matrizZeros(pesos.size(), pesos[0].size());
            m_bias = zeros(bias.size());
            v_bias = zeros(bias.size());
        }

        float fator1 = 1.0f - pow(b1, t);
        float fator2 = 1.0f - pow(b2, t);
        float umMenosB1 = 1.0f - b1;
        float umMenosB2 = 1.0f - b2;

        // FIX: decaimento de peso separado do Adam — sem L2 duplo
        for(size_t i = 0; i < pesos.size(); i++) {
            for(size_t j = 0; j < pesos[i].size(); j++) {
                // 1) decaimento de peso (AdamW)
                pesos[i][j] *= (1.0f - taxa * pesoDecaimento);
                // 2) passo Adam com gradiente puro
                float g = gradPesos[i][j];
                m_pesos[i][j] = b1 * m_pesos[i][j] + umMenosB1 * g;
                v_pesos[i][j] = b2 * v_pesos[i][j] + umMenosB2 * g * g;
                float mChapeu = m_pesos[i][j] / fator1;
                float vChapeu = v_pesos[i][j] / fator2;
                pesos[i][j] -= taxa * mChapeu / (sqrt(vChapeu) + eps);
            }
        }
        // bias sem decaimento de peso (padrão AdamW)
        for(size_t i = 0; i < bias.size(); i++) {
            float g = gradBias[i];
            m_bias[i] = b1 * m_bias[i] + umMenosB1 * g;
            v_bias[i] = b2 * v_bias[i] + umMenosB2 * g * g;
            float mChapeu = m_bias[i] / fator1;
            float vChapeu = v_bias[i] / fator2;
            bias[i] -= taxa * mChapeu / (sqrt(vChapeu) + eps);
        }
    }
};