// biblis/otimizadores.h
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
        pesos = attPesosAdam(pesos, gradPesos, m_pesos, v_pesos, taxa, b1, b2, eps, t);
        bias = attPesosAdam1D(bias, gradBias, m_bias, v_bias, taxa, b1, b2, eps, t);
    }
};

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

class AdaDelta : public Otimizador {
public:
    float taxa;
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
                // atualiza acumulador de gradientes
                acumGradPesos[i][j] = rho * acumGradPesos[i][j] + 
                (1.0f - rho) * gradPesos[i][j] * gradPesos[i][j];
                
                // calcula delta
                float delta = sqrt(acumDeltaPesos[i][j] + eps) / 
                             sqrt(acumGradPesos[i][j] + eps) * gradPesos[i][j];
                
                // atualiza pesos
                pesos[i][j] -= delta;
                
                // atualiza acumulador de deltas
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
        
        // aplica momentum primeiro
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
// adam com decaimento de peso correto
class AdamW : public Adam {
public:
    float pesoDecaimento;
    
    AdamW(float taxa = 0.001f, float pesoDecaimento = 0.01f)
        : Adam(taxa), pesoDecaimento(pesoDecaimento) {}
    
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
        // aplica decaimento de peso primeiro(AdamW)
        for(size_t i = 0; i < pesos.size(); i++) {
            for(size_t j = 0; j < pesos[i].size(); j++) {
                pesos[i][j] -= taxa * pesoDecaimento * pesos[i][j];
            }
        }
        // usa Adam normal
        pesos = attPesosAdam(pesos, gradPesos, m_pesos, v_pesos, taxa, b1, b2, eps, t);
        bias = attPesosAdam1D(bias, gradBias, m_bias, v_bias, taxa, b1, b2, eps, t);
    }
};