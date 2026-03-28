// biblis/otimis/adam.h
#pragma once
#include "otimizador.h"

class Adam : public Otimizador {
public:
    float taxa, b1, b2, eps;
    int t = 0;
    // cache pros momentos(m e v)
    vector<vector<float>> m_pesos, v_pesos;
    vector<float> m_bias, v_bias;

    Adam(float taxa = 0.001f) : taxa(taxa), b1(0.9f), b2(0.999f), eps(1e-8f) {}

    void att(vector<vector<float>>& pesos,
    vector<vector<float>>& gradPesos,
    vector<float>& bias,
    vector<float>& gradBias) override {
        t++;
        // inicia o cache se estiver vazio
        if(m_pesos.empty()) {
            m_pesos = matrizZeros(pesos.size(), pesos[0].size());
            v_pesos = matrizZeros(pesos.size(), pesos[0].size());
            m_bias = zeros(bias.size());
            v_bias = zeros(bias.size());
        }
        // FIX: sem lambda oculto — Adam puro
        pesos = attPesosAdam(pesos, gradPesos, m_pesos, v_pesos, taxa, b1, b2, eps, t);
        bias = attPesosAdam1D(bias, gradBias, m_bias, v_bias, taxa, b1, b2, eps, t);
    }
};