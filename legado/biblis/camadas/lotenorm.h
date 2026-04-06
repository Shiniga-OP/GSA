// biblis/camadas/lotenorm.h
#pragma once

#include "camada.h"

class LoteNorm : public Camada {
public:
    size_t numCaracteristicas;
    float epsilon;
    float momentum;
    bool treinando;
    
    // parametros treinaveis
    vector<float> gamma; // escala
    vector<float> beta; // deslocamento
    vector<float> gradGamma;
    vector<float> gradBeta;
    
    // estatisticas moveis(pra modo de inferencia)
    vector<float> mediaMovel;
    vector<float> varianciaMovel;
    
    // cache para retropropagação(pra lotes)
    vector<vector<float>> entradaCache;
    vector<vector<float>> entradaNormalizadaCache;
    vector<float> mediaCache;
    vector<float> varianciaCache;
    
    LoteNorm(size_t numCaracteristicas, float epsilon = 1e-5f, 
    float momentum = 0.9f, const string& nome = "")
    : Camada(nome), numCaracteristicas(numCaracteristicas), 
    epsilon(epsilon), momentum(momentum), treinando(true) {
        // inicia gamma com 1 e beta com 0
        gamma = vector<float>(numCaracteristicas, 1.0f);
        beta = vector<float>(numCaracteristicas, 0.0f);
        
        // inicia gradientes
        gradGamma = vector<float>(numCaracteristicas, 0.0f);
        gradBeta = vector<float>(numCaracteristicas, 0.0f);
        
        // inicia estatisticas moveis
        mediaMovel = vector<float>(numCaracteristicas, 0.0f);
        varianciaMovel = vector<float>(numCaracteristicas, 1.0f);
        
        tipo = "LoteNorm";
    }
    
    vector<float> prop(const vector<float>& entrada) override {
        if(entrada.size() != numCaracteristicas) {
            throw invalid_argument("[" + nome + "]: Dimensão de entrada incorreta para LoteNorm");
        }
        // pra um unico exemplo, tratamos como lote de tamanho 1
        vector<vector<float>> lote = {entrada};
        auto saidaLote = propLote(lote);
        return saidaLote[0];
    }
    
    vector<vector<float>> propLote(const vector<vector<float>>& entrada) override {
        if(entrada.empty()) return {};
        
        size_t loteTam = entrada.size();
        vector<vector<float>> saida(loteTam, vector<float>(numCaracteristicas));
        
        // limpa cache anterior
        entradaCache.clear();
        entradaNormalizadaCache.clear();
        
        if(treinando) {
            // modo de treino: calcula estatisticas sobre todo o lote
            // calcula media por caracteristica
            mediaCache = vector<float>(numCaracteristicas, 0.0f);
            for(size_t j = 0; j < numCaracteristicas; j++) {
                for(size_t i = 0; i < loteTam; i++) {
                    mediaCache[j] += entrada[i][j];
                }
                mediaCache[j] /= loteTam;
            }
            // calcula variancia por caracteristica
            varianciaCache = vector<float>(numCaracteristicas, 0.0f);
            for(size_t j = 0; j < numCaracteristicas; j++) {
                for(size_t i = 0; i < loteTam; i++) {
                    float diff = entrada[i][j] - mediaCache[j];
                    varianciaCache[j] += diff * diff;
                }
                varianciaCache[j] /= loteTam;
            }
            // armazena entrada pra retropropagação
            entradaCache = entrada;
            
            // normaliza e aplica transformação afim
            entradaNormalizadaCache.resize(loteTam, vector<float>(numCaracteristicas));
            for(size_t i = 0; i < loteTam; i++) {
                for(size_t j = 0; j < numCaracteristicas; j++) {
                    float desvio = sqrt(varianciaCache[j] + epsilon);
                    float norm = (entrada[i][j] - mediaCache[j]) / desvio;
                    entradaNormalizadaCache[i][j] = norm;
                    saida[i][j] = gamma[j] * norm + beta[j];
                }
            }
            // atualiza estatisticas moveis
            for(size_t j = 0; j < numCaracteristicas; j++) {
                mediaMovel[j] = momentum * mediaMovel[j] + (1.0f - momentum) * mediaCache[j];
                varianciaMovel[j] = momentum * varianciaMovel[j] + (1.0f - momentum) * varianciaCache[j];
            }
        } else {
            // modo inferencia: usa estatisticas moveis
            for(size_t i = 0; i < loteTam; i++) {
                for(size_t j = 0; j < numCaracteristicas; j++) {
                    float norm = (entrada[i][j] - mediaMovel[j]) / 
                    sqrt(varianciaMovel[j] + epsilon);
                    saida[i][j] = gamma[j] * norm + beta[j];
                }
            }
        }
        return saida;
    }
    
    GradGenerico retroprop(const vector<float>& gradiente) override {
        vector<vector<float>> gradLote = {gradiente};
        auto gradEntradaLote = retropropLote(gradLote);
        return GradGenerico(gradEntradaLote[0]);
    }
    
    vector<vector<float>> retropropLote(const vector<vector<float>>& gradiente) override {
        size_t loteTam = gradiente.size();
        vector<vector<float>> dEntrada(loteTam, vector<float>(numCaracteristicas, 0.0f));
        
        // modo inferencia: apenas passa o gradiente(não ha estatisticas de lote)
        if(!treinando) return gradiente;
        
        if(entradaCache.size() != loteTam) {
            throw runtime_error("[" + nome + "]: Tamanho do lote não corresponde ao cache");
        }
        // calcula gradientes para gamma e beta
        for(size_t j = 0; j < numCaracteristicas; j++) {
            float somaGamma = 0.0f;
            float somaBeta = 0.0f;
            
            for(size_t i = 0; i < loteTam; i++) {
                somaGamma += gradiente[i][j] * entradaNormalizadaCache[i][j];
                somaBeta += gradiente[i][j];
            }
            gradGamma[j] += somaGamma;
            gradBeta[j] += somaBeta;
        }
        // calcula gradiente em relação a entrada
        float n = static_cast<float>(loteTam);
        
        for(size_t j = 0; j < numCaracteristicas; j++) {
            // calcula somas pra essa caracteristica
            float somaGrad = 0.0f;
            float somaGradX = 0.0f;
            
            for(size_t i = 0; i < loteTam; i++) {
                float grad = gradiente[i][j] * gamma[j];
                float norm = entradaNormalizadaCache[i][j];
                
                somaGrad += grad;
                somaGradX += grad * norm;
            }
            // calcula gradiente pra cada exemplo
            float desvio = sqrt(varianciaCache[j] + epsilon);
            
            for(size_t i = 0; i < loteTam; i++) {
                float grad = gradiente[i][j] * gamma[j];
                float norm = entradaNormalizadaCache[i][j];
                
                dEntrada[i][j] = (1.0f / desvio) * (grad - (1.0f / n)
                * somaGrad - (1.0f / n) * norm * somaGradX);
            }
        }
        return dEntrada;
    }
    
    // usa retropropLote quando disponivel
    void att(float taxaAprendizado) override {
        if(otimizador) {
            // converte ambos pra 2D, passa pro otimizador e le os dois de volta
            vector<vector<float>> gammaMat = {gamma};
            vector<vector<float>> gradGammaMat = {gradGamma};
            vector<vector<float>> betaMat = {beta};
            vector<vector<float>> gradBetaMat = {gradBeta};
            
            // gamma como "pesos", beta como "bias" ambos atualizados corretamente
            otimizador->att(gammaMat, gradGammaMat, betaMat[0], gradBetaMat[0]);
            
            // le os dois de volta
            gamma = gammaMat[0];
            beta = betaMat[0];
        } else {
            // atualização SGD padrão
            for(size_t i = 0; i < numCaracteristicas; i++) {
                gamma[i] -= taxaAprendizado * gradGamma[i];
                beta[i] -= taxaAprendizado * gradBeta[i];
            }
        }
    }
    
    void zerarGradientes() override {
        fill(gradGamma.begin(), gradGamma.end(), 0.0f);
        fill(gradBeta.begin(), gradBeta.end(), 0.0f);
    }
    
    // info
    bool temParametros() const override { return true; }
    // gamma + beta
    size_t numParametros() const override { return 2 * numCaracteristicas; }
    
    // serialização
    void salvar(const string& nomeArquivo) const override {
        ofstream arquivo(nomeArquivo);
        if(!arquivo) throw runtime_error("[" + nome + "]: Não foi possível salvar LoteNorm");
        
        arquivo << "LOTENORM_CAMADA" << endl;
        arquivo << numCaracteristicas << " " << epsilon << " " << momentum << endl;
        
        // salva gamma
        for(float g : gamma) arquivo << g << " ";
        arquivo << endl;
        
        // salva beta
        for(float b : beta) arquivo << b << " ";
        arquivo << endl;
        
        // salva estatisticas moveis
        for(float m : mediaMovel) arquivo << m << " ";
        arquivo << endl;
        
        for(float v : varianciaMovel) arquivo << v << " ";
        arquivo << endl;
        
        arquivo.close();
    }
    
    void carregar(const string& nomeArquivo) override {
        ifstream arquivo(nomeArquivo);
        if(!arquivo) throw runtime_error("[" + nome + "]: Não foi possível carregar LoteNorm");
        
        string tipo;
        arquivo >> tipo;
        if(tipo != "LOTENORM_CAMADA") {
            throw runtime_error("[" + nome + "]: Formato de arquivo inválido");
        }
        arquivo >> numCaracteristicas >> epsilon >> momentum;
        
        // redimensiona vetores
        gamma.resize(numCaracteristicas);
        beta.resize(numCaracteristicas);
        gradGamma.resize(numCaracteristicas);
        gradBeta.resize(numCaracteristicas);
        mediaMovel.resize(numCaracteristicas);
        varianciaMovel.resize(numCaracteristicas);
        
        // carrega gamma
        for(size_t i = 0; i < numCaracteristicas; i++) {
            arquivo >> gamma[i];
        }
        // carrega beta
        for(size_t i = 0; i < numCaracteristicas; i++) {
            arquivo >> beta[i];
        }
        // carrega estatísticas móveis
        for(size_t i = 0; i < numCaracteristicas; i++) {
            arquivo >> mediaMovel[i];
        }
        for(size_t i = 0; i < numCaracteristicas; i++) {
            arquivo >> varianciaMovel[i];
        }
        arquivo.close();
    }
};