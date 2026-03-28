// biblis/otimis/otimizador.h
#pragma once
class Otimizador {
public:
    virtual ~Otimizador() = default;
    virtual void att(
        vector<vector<float>>& pesos,
        vector<vector<float>>& gradPesos,
        vector<float>& bias,
        vector<float>& gradBias
    ) = 0;
};
