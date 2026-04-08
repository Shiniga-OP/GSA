struct Otimizador {
    virtual void att(
        float* pesos, float* bias,
        float* gradPesos, float* gradBias
    ) = 0;
    virtual ~Otimizador() = default;
};