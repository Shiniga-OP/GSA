// biblis/camadas.h
#pragma once
#include <vector>
#include <memory>
#include <string>
#include <functional>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include "ativas.h"
#include "util.h"
#include "otimizadores.h"

using namespace std;

class Camada {
public:
    string tipo;
    string nome;
    unique_ptr<Otimizador> otimizador;
    
    Camada(const string& nome = "") : nome(nome) {}
    virtual ~Camada() = default;
    
    virtual vector<float> prop(const vector<float>& entrada) = 0; // propagação
    virtual vector<float> retroprop(const vector<float>& gradiente) = 0; // retropropagação
    
    // lotes:
    virtual vector<vector<float>> propLote(const vector<vector<float>>& entrada) {
        // processa cada exemplo sozinho
        vector<vector<float>> saida;
        for(const auto& e : entrada) {
            saida.push_back(prop(e));
        }
        return saida;
    }
    virtual vector<vector<float>> retropropLote(const vector<vector<float>>& gradiente) {
        vector<vector<float>> resultado;
        for(const auto& g : gradiente) {
            resultado.push_back(retroprop(g));
        }
        return resultado;
    }
    virtual void att(float taxaAprendizado) = 0;
    virtual void zerarGradientes() = 0;
    
    // otimizadores:
    virtual void defOtimizador(unique_ptr<Otimizador> otim) {
        otimizador = std::move(otim);
    }
    // pra camadas treinaveis
    virtual bool temParametros() const { return false; }
    virtual size_t numParametros() const { return 0; }
    
    // serialização
    virtual void salvar(const string& arquivo) const = 0;
    virtual void carregar(const string& arquivo) = 0;
};

class Densa : public Camada {
public:
    size_t entradaDim;
    size_t saidaDim;
    vector<vector<float>> pesos; // [saida x entrada]
    vector<float> bias; // [saida]
    vector<vector<float>> gradPesos; // gradientes dos pesos
    vector<float> gradBias; // gradientes do bias
    
    function<float(float)> ativacao;
    function<float(float)> derivadaAtivacao;
    
    // cache pra prop/retroprop
    vector<float> entradaCache;
    vector<float> ativacaoCache;
    
    bool usarBias;
    string tipoAtivacao;
    
    // construtores
    Densa(size_t entradaDim, size_t saidaDim, 
    const string& tipoAtivacao = "linear",
    bool usarBias = true,
    const string& nome = "")
    : Camada(nome), entradaDim(entradaDim), saidaDim(saidaDim), 
    usarBias(usarBias), tipoAtivacao(tipoAtivacao) {
        
        // inicia os pesos(He/Xavier baseado na ativação)
        if(tipoAtivacao == "relu" || tipoAtivacao == "leakyrelu") {
            pesos = iniPesosHe(saidaDim, entradaDim);
        } else {
            pesos = iniPesosXavier(saidaDim, entradaDim);
        }
        bias = zeros(saidaDim);
        
        // inicia os gradientes
        gradPesos = vector<vector<float>>(saidaDim, vector<float>(entradaDim, 0.0f));
        gradBias = zeros(saidaDim);
        
        // config função de ativação
        configAtivacao(tipoAtivacao);
        tipo = "Densa";
    }
    
    void configAtivacao(const string& tipo) {
        tipoAtivacao = tipo;
        
        if(tipo == "sigmoid") {
            ativacao = sigmoid;
            derivadaAtivacao = [](float y) { return y * (1 - y); };
        } else if(tipo == "relu") {
            ativacao = ReLU;
            derivadaAtivacao = [](float y) { return y > 0 ? 1.0f : 0.0f; };
        } else if(tipo == "leakyrelu") {
            ativacao = leakyReLU;
            derivadaAtivacao = derivadaLeakyReLU;
        } else if(tipo == "tanh") {
            ativacao = tanhF;
            derivadaAtivacao = derivadaTanh;
        } else if(tipo == "softmax") {
            // softmax é especial é tratado separadamente
            ativacao = nullptr;
            derivadaAtivacao = nullptr;
        } else { // linear(sem ativação)
            ativacao = [](float x) { return x; };
            derivadaAtivacao = [](float y) { return 1.0f; };
        }
    }
    
    vector<float> prop(const vector<float>& entrada) override {
        if(entrada.size() != entradaDim) {
            throw invalid_argument("[" + nome + "]: Dimensão de entrada incorreta para camada densa");
        }
        // cache da entrada
        entradaCache = entrada;
        
        // calcular z = Px + b
        vector<float> z = aplicarMatriz(pesos, entrada);
        
        if(usarBias) {
            z = somarVetores(z, bias);
        }
        // aplica ativação
        vector<float> saida(z.size());
        if(tipoAtivacao == "softmax") {
            saida = softmax(z);
        } else if(ativacao) {
            for(size_t i = 0; i < z.size(); i++) {
                saida[i] = ativacao(z[i]);
            }
        } else {
            saida = z; // linear
        }
        // cache da ativação(ou z se for softmax)
        ativacaoCache = saida;
        
        return saida;
    }
    
    vector<vector<float>> propLote(const vector<vector<float>>& entrada) override {
        vector<vector<float>> saida;
        
        // processar cada exemplo sozinho
        for(const auto& e : entrada) {
            saida.push_back(prop(e));
        }
        return saida;
    }
    
    vector<float> retroprop(const vector<float>& gradiente) override {
        if(gradiente.size() != saidaDim) {
            throw invalid_argument("[" + nome + "]: Dimensão do gradiente incorreta");
        }
        // gradiente em relação a ativação
        vector<float> gradAtivacao = gradiente;
        
        // se houver ativação(exceto softmax), aplica derivada
        if(tipoAtivacao != "softmax" && tipoAtivacao != "linear" && derivadaAtivacao) {
            for(size_t i = 0; i < gradAtivacao.size(); i++) {
                gradAtivacao[i] *= derivadaAtivacao(ativacaoCache[i]);
            }
        }
        // calcula gradientes dos pesos: dP = grad * entrada^T
        for(size_t i = 0; i < saidaDim; i++) {
            for(size_t j = 0; j < entradaDim; j++) {
                gradPesos[i][j] += gradAtivacao[i] * entradaCache[j];
            }
        }
        // gradiente do bias
        if(usarBias) {
            for(size_t i = 0; i < saidaDim; i++) {
                gradBias[i] += gradAtivacao[i];
            }
        }
        // gradiente pra camada anterior: dE/dx = P^T * grad
        vector<float> gradEntrada(entradaDim, 0.0f);
        for(size_t j = 0; j < entradaDim; j++) {
            for(size_t i = 0; i < saidaDim; i++) {
                gradEntrada[j] += pesos[i][j] * gradAtivacao[i];
            }
        }
        return gradEntrada;
    }
    
    void att(float taxaAprendizado) override {
        if(otimizador) {
            otimizador->att(pesos, gradPesos, bias, gradBias);
        } else {
            // atualiza pesos
            for(size_t i = 0; i < saidaDim; i++) {
                for(size_t j = 0; j < entradaDim; j++) {
                    pesos[i][j] -= taxaAprendizado * gradPesos[i][j];
                }
            }
            // atualiza bias
            if(usarBias) {
                for(size_t i = 0; i < saidaDim; i++) {
                    bias[i] -= taxaAprendizado * gradBias[i];
                }
            }
        }
    }
    
    void zerarGradientes() override {
        // zera gradientes dos pesos
        for(auto& linha : gradPesos) {
            fill(linha.begin(), linha.end(), 0.0f);
        }
        // zera gradientes do bias
        fill(gradBias.begin(), gradBias.end(), 0.0f);
    }
    
    void defPesos(const vector<vector<float>>& novosPesos) {
        if(novosPesos.size() != saidaDim || novosPesos[0].size() != entradaDim) {
            throw invalid_argument("[" + nome + "]: Dimensões dos pesos incorretas");
        }
        pesos = novosPesos;
    }
    
    void defBias(const vector<float>& novoBias) {
        if(novoBias.size() != saidaDim) {
            throw invalid_argument("[" + nome + "]: Dimensão do bias incorreta");
        }
        bias = novoBias;
    }
    
    // informações da camada
    bool temParametros() const override { return true; }
    size_t numParametros() const override { 
        return saidaDim * entradaDim + (usarBias ? saidaDim : 0);
    }
    
    // serialização
    void salvar(const string& nomeArquivo) const override {
        ofstream arquivo(nomeArquivo);
        if(!arquivo) throw runtime_error("[" + nome + "]: Não foi possível salvar a camada");
        
        arquivo << "DENSA_CAMADA" << endl;
        arquivo << entradaDim << " " << saidaDim << endl;
        arquivo << tipoAtivacao << " " << (usarBias ? 1 : 0) << endl;
        
        // salva pesos
        for(const auto& linha : pesos) {
            for(float p : linha) arquivo << p << " ";
            arquivo << endl;
        }
        // salva bias
        for(float b : bias) arquivo << b << " ";
        arquivo << endl;
        
        arquivo.close();
    }
    
    void carregar(const string& nomeArquivo) override {
        ifstream arquivo(nomeArquivo);
        if(!arquivo) throw runtime_error("[" + nome + "]: Não foi possível carregar a camada");
        
        string tipo;
        arquivo >> tipo;
        if(tipo != "DENSA_CAMADA") {
            throw runtime_error("[" + nome + "]: Formato de arquivo inválido");
        }
        arquivo >> entradaDim >> saidaDim;
        
        int usarBiasInt;
        arquivo >> tipoAtivacao >> usarBiasInt;
        usarBias = (usarBiasInt == 1);
        
        configAtivacao(tipoAtivacao);
        
        // carrega pesos
        pesos = vector<vector<float>>(saidaDim, vector<float>(entradaDim, 0.0f));
        for(size_t i = 0; i < saidaDim; i++) {
            for(size_t j = 0; j < entradaDim; j++) {
                arquivo >> pesos[i][j];
            }
        }
        // carrega bias
        bias = vector<float>(saidaDim, 0.0f);
        for(size_t i = 0; i < saidaDim; i++) {
            arquivo >> bias[i];
        }
        arquivo.close();
        
        // reinicia gradientes com dimensões certas
        gradPesos = vector<vector<float>>(saidaDim, vector<float>(entradaDim, 0.0f));
        gradBias = vector<float>(saidaDim, 0.0f);
    }
};
// camada de dropout:
class Dropout : public Camada {
public:
    float taxa;
    vector<bool> mascara;
    bool treinando;
    mt19937 gen;
    bernoulli_distribution dist;
    
    Dropout(float taxa = 0.5f, const string& nome = "", int seed = 42) 
        : Camada(nome), taxa(taxa), treinando(true), 
          dist(1.0f - taxa) {  // distribuição pré-calculada
        
        if(taxa < 0.0f || taxa >= 1.0f) {
            throw invalid_argument("[" + nome + "]: Taxa de dropout deve estar em [0, 1)");
        }
        tipo = "Dropout";
        gen.seed(seed);
    }
    
    vector<float> prop(const vector<float>& entrada) override {
        vector<float> saida = entrada; // começa com copia
        
        if(treinando && taxa > 0.0f) {
            // gera nova mascara pra essa propagação
            mascara.resize(entrada.size());
            
            for(size_t i = 0; i < entrada.size(); i++) {
                mascara[i] = dist(gen); // true = mantem, false = dropa
                if(!mascara[i]) {
                    saida[i] = 0.0f;
                } else {
                    saida[i] /= (1.0f - taxa); // escalonamento
                }
            }
        }
        // se não estiver treinando ou taxa = 0, saida = entrada
        return saida;
    }
    
    vector<vector<float>> propLote(const vector<vector<float>>& entrada) override {
        vector<vector<float>> saida;
        saida.reserve(entrada.size());
        
        for(const auto& e : entrada) {
            saida.push_back(prop(e));
        }
        return saida;
    }
    
    vector<float> retroprop(const vector<float>& gradiente) override {
        if(!treinando || taxa == 0.0f) {
            return gradiente;  // modo teste: passa tudo
        }
        // modo treino: aplica a mesma mascara da propagação
        if(mascara.size() != gradiente.size()) {
            throw std::runtime_error("[" + nome + "]: Máscara não gerada na propagação");
        }
        vector<float> gradEntrada(gradiente.size());
        
        for(size_t i = 0; i < gradiente.size(); i++) {
            if(mascara[i]) {
                gradEntrada[i] = gradiente[i] / (1.0f - taxa);
            } else {
                gradEntrada[i] = 0.0f;
            }
        }
        return gradEntrada;
    }
    // dropout não tem parametros pra atualizar
    void att(float taxaAprendizado) override {}
    // dropout não tem gradientes    
    void zerarGradientes() override {}
    
    bool temParametros() const override { return false; }
    size_t numParametros() const override { return 0; }
    
    void salvar(const string& arquivoNome) const override {
        ofstream arquivo(arquivoNome);
        if(!arquivo) throw runtime_error("Não foi possível salvar Dropout");
        
        arquivo << "DROPOUT_CAMADA" << endl;
        arquivo << taxa << endl;
        
        arquivo.close();
    }
    
    void carregar(const string& arquivoNome) override {
        ifstream arquivo(arquivoNome);
        if(!arquivo) throw runtime_error("Não foi possível carregar Dropout");
        
        string tipo;
        arquivo >> tipo;
        if(tipo != "DROPOUT_CAMADA") {
            throw runtime_error("Formato de arquivo inválido para Dropout");
        }
        arquivo >> taxa;
        dist = bernoulli_distribution(1.0f - taxa);  // recalcula distribuição
        
        arquivo.close();
    }
};
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
    
    vector<float> retroprop(const vector<float>& gradiente) override {
        // pra compatibilidade, tratamos como lote de tamanho 1
        vector<vector<float>> gradLote = {gradiente};
        auto gradEntradaLote = retropropLote(gradLote);
        
        // retorna apenas o primeiro exemplo(unico)
        return gradEntradaLote[0];
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
            // converte vetores 1D pra 2D pro otimizador
            vector<vector<float>> gammaMat = {gamma};
            vector<vector<float>> gradGammaMat = {gradGamma};
            vector<vector<float>> betaMat = {beta};
            vector<vector<float>> gradBetaMat = {gradBeta};
            
            otimizador->att(gammaMat, gradGammaMat, beta, gradBeta);
            
            // Atualiza gamma e beta
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
// modelo pra gerenciamento de camadas:
class Modelo {
public:
    vector<unique_ptr<Camada>> camadas;
    string nome;
    bool treinando;
    
    Modelo(const string& nome = "") : nome(nome), treinando(true) {}
    
    void add(unique_ptr<Camada> camada) {
        camadas.push_back(std::move(camada));
    }
    
    // propagação
    vector<float> prop(const vector<float>& entrada) {
        vector<float> resultado = entrada;
        
        for(const auto& camada : camadas) {
            resultado = camada->prop(resultado);
        }
        return resultado;
    }
    
    // propagação em lote
    vector<vector<float>> propLote(const vector<vector<float>>& entrada) {
        vector<vector<float>> resultado = entrada;
        
        for(const auto& camada : camadas) {
            resultado = camada->propLote(resultado);
        }
        return resultado;
    }
    
    // retropropagação atraves do modelo inteiro
    vector<float> retroprop(const vector<float>& gradiente) {
        vector<float> grad = gradiente;
        
        // retropropaga na ordem inversa
        for(int i = camadas.size() - 1; i >= 0; i--) {
            grad = camadas[i]->retroprop(grad);
        }
        return grad;
    }
    
    // atualização dos pesos
    void att(float taxaAprendizado) {
        for(auto& camada : camadas) {
            if(camada->temParametros()) {
                camada->att(taxaAprendizado);
            }
        }
    }
    
    void zerarGradientes() {
        for(auto& camada : camadas) {
            if(camada->temParametros()) {
                camada->zerarGradientes();
            }
        }
    }
    
    void modoTreino() {
        treinando = true;
        for(auto& camada : camadas) {
            auto* dropout = dynamic_cast<Dropout*>(camada.get());
            if(dropout) dropout->treinando = true;
            
            auto* lotenorm = dynamic_cast<LoteNorm*>(camada.get());
            if(lotenorm) lotenorm->treinando = true;
        }
    }
    
    void modoTeste() {
        treinando = false;
        for(auto& camada : camadas) {
            auto* dropout = dynamic_cast<Dropout*>(camada.get());
            if(dropout) dropout->treinando = false;
            
            auto* lotenorm = dynamic_cast<LoteNorm*>(camada.get());
            if(lotenorm) lotenorm->treinando = false;
        }
    }
    
    float treinar(const vector<float>& entrada, const vector<float>& alvo, 
    function<float(const vector<float>&, const vector<float>&)> perda,
    float taxaAprendizado = 0.01f) {
        // garante que ta no modo treino
        modoTreino();
        
        // propagação
        vector<float> saida = prop(entrada);
        
        // calcula a perda
        float erro = perda(saida, alvo);
        
        // gradiente inicial(derivada da MSE)
        vector<float> gradiente(saida.size());
        for(size_t i = 0; i < saida.size(); i++) {
            gradiente[i] = 2.0f * (saida[i] - alvo[i]) / saida.size();
        }
        // repropagação
        for(int i = camadas.size() - 1; i >= 0; i--) {
            gradiente = camadas[i]->retroprop(gradiente);
        }
        // atualiza pesos(apenas camadas com parametros)
        for(auto& camada : camadas) {
            if(camada->temParametros()) {
                camada->att(taxaAprendizado);
            }
        }
        // zera gradientes(apenas camadas com parametros)
        for(auto& camada : camadas) {
            if(camada->temParametros()) {
                camada->zerarGradientes();
            }
        }
        return erro;
    }
    
    // info
    size_t numParametros() const {
        size_t total = 0;
        for(const auto& camada : camadas) {
            total += camada->numParametros();
        }
        return total;
    }
    
    void resumo() const {
        cout << "=== Modelo: " << nome << " ===" << endl;
        cout << "Numero de camadas: " << camadas.size() << endl;
        cout << "Total de parametros: " << numParametros() << endl;
        cout << "Camadas:" << endl;
        
        for(size_t i = 0; i < camadas.size(); i++) {
            cout << "  [" << i << "] " << camadas[i]->nome 
            << " (" << camadas[i]->tipo << ")" 
            << " - Parametros: " << camadas[i]->numParametros() << endl;
        }
    }
};