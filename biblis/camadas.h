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
        vector<vector<float>> res;
        for(const auto& g : gradiente) {
            res.push_back(retroprop(g));
        }
        return res;
    }
    // mapas:
    virtual vector<vector<vector<float>>> propMapa(const vector<vector<vector<float>>>& entrada) {
        throw runtime_error("[" + nome + "]: Método propMapa não implementado");
    }
    virtual vector<vector<vector<float>>> retropropMapa(const vector<vector<vector<float>>>& gradiente) {
        throw runtime_error("[" + nome + "]: Método retropropMapa não implementado");
    }
    virtual vector<vector<vector<vector<float>>>> propLoteMapa(const vector<vector<vector<vector<float>>>>& entrada) {
        vector<vector<vector<vector<float>>>> saida;
        for(const auto& e : entrada) {
            saida.push_back(propMapa(e));
        }
        return saida;
    }
    // pesos e gradientes:
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
        
        // calcula z = Px + b
        vector<float> z = aplicarMatriz(pesos, entrada);
        
        if(usarBias) {
            z = somarVetores(z, bias);
        }
        // aplica ativação
        vector<float> saida(z.size());
        if(tipoAtivacao == "softmax") {
            saida = softmax(z);
            // armazena tanto os logits(z) quanto a saida softmax
            ativacaoCache = saida; // armazena a saida softmax
        } else if(ativacao) {
            for(size_t i = 0; i < z.size(); i++) {
                saida[i] = ativacao(z[i]);
            }
            ativacaoCache = saida; // armazena a ativação pra outras funções
        } else {
            saida = z; // linear
            ativacaoCache = saida;
        }
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
        
        vector<float> gradAtivacao = gradiente;
        
        if(tipoAtivacao == "softmax") {
            // ativacaoCache = saida do softmax e gradiente
            gradAtivacao = derivadaSoftmax(ativacaoCache, gradiente);
            
        } else if(tipoAtivacao != "linear" && derivadaAtivacao) {
            // outras ativações aplica derivada normalmente
            for(size_t i = 0; i < gradAtivacao.size(); i++) {
                gradAtivacao[i] *= derivadaAtivacao(ativacaoCache[i]);
            }
        }
        // pra "linear", não faz nada(derivada = 1)
        
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
    
    vector<vector<vector<float>>> propMapa(const vector<vector<vector<float>>>& entrada3D) override {
        // converte de 3D pra 1D
        vector<float> entrada1D;
        for(const auto& canal : entrada3D) {
            for(const auto& linha : canal) {
                for(float pixel : linha) {
                    entrada1D.push_back(pixel);
                }
            }
        }
        // processa
        auto saida1D = prop(entrada1D);
        
        // converte de volta pra 3D(1 canal)
        vector<vector<vector<float>>> saida3D(1);
        saida3D[0].resize(1);
        saida3D[0][0] = saida1D;
        
        return saida3D;
    }
    
    vector<vector<vector<float>>> retropropMapa(const vector<vector<vector<float>>>& gradiente) override {
        // converte gradiente 3D pra 1D
        vector<float> gradiente1D;
        for(const auto& canal : gradiente) {
            for(const auto& linha : canal) {
                for(float pixel : linha) {
                    gradiente1D.push_back(pixel);
                }
            }
        }
        // retropropaga
        auto gradEntrada1D = retroprop(gradiente1D);
        
        // converte de volta pra 3D(1 canal)
        vector<vector<vector<float>>> gradEntrada3D(1);
        gradEntrada3D[0].resize(1);
        gradEntrada3D[0][0] = gradEntrada1D;
        
        return gradEntrada3D;
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
// camada de normalização em lote:
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
// camada convolucional 2D:
class Conv2D : public Camada {
public:
    size_t filtros; // numero de filtros
    size_t canalEntrada; // canais de entrada(1 para escala cinza, 3 pra RGB)
    size_t alturaFiltro; // altura do kernel/filtro
    size_t larguraFiltro; // largura do kernel/filtro
    size_t passo; // passo da convolução
    size_t espaco; // espaco(0 ou 1)
    
    // parametros treinaveis
    vector<vector<vector<vector<float>>>> pesos; // [filtros][canal][altura][largura]
    vector<float> bias; // [filtros]
    
    // gradientes
    vector<vector<vector<vector<float>>>> gradPesos;
    vector<float> gradBias;
    
    // cache pra retropropagação
    vector<vector<vector<float>>> entradaCache; // entrada da camada
    vector<vector<vector<float>>> saidaCache; // saida da camada(apos convolução)
    
    // dimensões da entrada e saida
    size_t entradaAltura, entradaLargura;
    size_t saidaAltura, saidaLargura;
    
    // função de ativação
    string tipoAtivacao;
    function<float(float)> ativacao;
    function<float(float)> derivadaAtivacao;
    
    bool usarBias;
    
    Conv2D(size_t filtros, size_t alturaFiltro, 
    size_t larguraFiltro, size_t canalEntrada = 1,
    size_t passo = 1, size_t espaco = 0,
    const string& tipoAtivacao = "relu", bool usarBias = true,
    const string& nome = "")
    : Camada(nome), filtros(filtros),
    alturaFiltro(alturaFiltro), larguraFiltro(larguraFiltro),
    canalEntrada(canalEntrada), passo(passo),
    espaco(espaco), usarBias(usarBias), tipoAtivacao(tipoAtivacao) {
        tipo = "Conv2D";
        
        // inicia os pesos com He(boa pra ReLU)
        iniciarPesos();
        
        // inicia bias com zeros
        bias = zeros(filtros);
        
        // inicia gradientes
        iniciarGrad();
        
        // cobfigura a função de ativação
        configAtivacao(tipoAtivacao);
    }
    
    void iniciarPesos() {
        pesos.resize(filtros);
        
        // fator de escala pra He
        float escala = sqrt(2.0f / (alturaFiltro * larguraFiltro * canalEntrada));
        
        random_device al;
        mt19937 gen(al());
        normal_distribution<float> dist(0.0f, escala);
        
        for(size_t f = 0; f < filtros; f++) {
            pesos[f].resize(canalEntrada);
            
            for(size_t c = 0; c < canalEntrada; c++) {
                pesos[f][c].resize(alturaFiltro);
                
                for(size_t i = 0; i < alturaFiltro; i++) {
                    pesos[f][c][i].resize(larguraFiltro);
                    
                    for(size_t j = 0; j < larguraFiltro; j++) {
                        pesos[f][c][i][j] = dist(gen);
                    }
                }
            }
        }
    }
    
    void iniciarGrad() {
        gradPesos = zeros4D(filtros, canalEntrada, alturaFiltro, larguraFiltro);
        gradBias = zeros(filtros);
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
        } else { // linear(sem ativação)
            ativacao = [](float x) { return x; };
            derivadaAtivacao = [](float y) { return 1.0f; };
        }
    }
    
    // calcula dimensões de saida
    void calcularDimensoesSaida(size_t alturaEntrada, size_t larguraEntrada) {
        saidaAltura = (alturaEntrada + 2 * espaco - alturaFiltro) / passo + 1;
        saidaLargura = (larguraEntrada + 2 * espaco - larguraFiltro) / passo + 1;
        
        if(saidaAltura <= 0 || saidaLargura <= 0) {
            throw invalid_argument("[" + nome + "]: Dimensões de saída inválidas. Verifique filtro, passo e espaco.");
        }
    }
    
    // aplica espaco em uma imagem
    vector<vector<vector<float>>> aplicarespaco(const vector<vector<vector<float>>>& entrada) const {
        if(espaco == 0) return entrada;
        
        size_t novaAltura = entradaAltura + 2 * espaco;
        size_t novaLargura = entradaLargura + 2 * espaco;
        
        vector<vector<vector<float>>> comespaco(canalEntrada, 
        vector<vector<float>>(novaAltura, vector<float>(novaLargura, 0.0f)));
        
        for(size_t c = 0; c < canalEntrada; c++) {
            for(size_t i = 0; i < entradaAltura; i++) {
                for(size_t j = 0; j < entradaLargura; j++) {
                    comespaco[c][i + espaco][j + espaco] = entrada[c][i][j];
                }
            }
        }
        return comespaco;
    }
    
    // realiza a operação de convolução pra um unico filtro
    vector<vector<float>> convoluirFiltro(
        const vector<vector<vector<float>>>& entradaComEspaco,
        size_t filtroIdc) const {
        
        vector<vector<float>> res(saidaAltura, vector<float>(saidaLargura, 0.0f));
        
        for(size_t y = 0; y < saidaAltura; y++) {
            for(size_t x = 0; x < saidaLargura; x++) {
                float soma = 0.0f;
                
                // percorre todos os canais
                for(size_t c = 0; c < canalEntrada; c++) {
                    // aplica o filtro nessa região
                    for(size_t i = 0; i < alturaFiltro; i++) {
                        for(size_t j = 0; j < larguraFiltro; j++) {
                            size_t entradaY = y * passo + i;
                            size_t entradaX = x * passo + j;
                            
                            soma += entradaComEspaco[c][entradaY][entradaX] * 
                            pesos[filtroIdc][c][i][j];
                        }
                    }
                }
                res[y][x] = soma;
            }
        }
        return res;
    }
    
    vector<float> prop(const vector<float>& entrada) override {
        throw runtime_error("[" + nome + "]: Use propMapa para Conv2D(entrada deve ser 3D)");
    }
    
    // propagação pra mapa de caracteristicas 2D
    vector<vector<vector<float>>> propMapa(const vector<vector<vector<float>>>& entrada) override {
        if(entrada.size() != canalEntrada) {
            throw invalid_argument("[" + nome + "]: Número de canais de entrada incorreto");
        }
        entradaAltura = entrada[0].size();
        entradaLargura = entrada[0][0].size();
        
        // calcula dimensões de saida
        calcularDimensoesSaida(entradaAltura, entradaLargura);
        
        // aplica espaco se necessario
        auto entradaComEspaco = aplicarespaco(entrada);
        
        // armazena entrada no cache
        entradaCache = entradaComEspaco;
        
        // cria tensor de saida [filtros][altura][largura]
        vector<vector<vector<float>>> saida(filtros);
        
        // pra cada filtro
        for(size_t f = 0; f < filtros; f++) {
            saida[f] = convoluirFiltro(entradaComEspaco, f);
            
            // adiciona bias
            if(usarBias) {
                for(auto& linha : saida[f]) {
                    for(auto& pixel : linha) {
                        pixel += bias[f];
                    }
                }
            }
            // aplica função de ativação
            for(auto& linha : saida[f]) {
                for(auto& pixel : linha) {
                    pixel = ativacao(pixel);
                }
            }
        }
        // armazena saida no cache
        saidaCache = saida;
        
        return saida;
    }
    
    // propagação em lote(multiplas imagens)
    vector<vector<vector<vector<float>>>> propLoteMapa(const vector<vector<vector<vector<float>>>>& entradaLote) override {
        vector<vector<vector<vector<float>>>> saidaLote;
        
        for(const auto& entrada : entradaLote) {
            saidaLote.push_back(propMapa(entrada));
        }
        return saidaLote;
    }
    
    vector<float> retroprop(const vector<float>& gradiente) override {
        throw runtime_error("[" + nome + "]: Use retropropMapa para Conv2D");
    }
    
    // retropropagação pra mapa de caracteristicas 2D
    vector<vector<vector<float>>> retropropMapa(const vector<vector<vector<float>>>& gradienteSaida) override {
        if(gradienteSaida.size() != filtros) {
            throw invalid_argument("[" + nome + "]: Dimensão do gradiente de saída incorreta");
        }
        size_t gradAltura = gradienteSaida[0].size();
        size_t gradLargura = gradienteSaida[0][0].size();
        
        if(gradAltura != saidaAltura || gradLargura != saidaLargura) {
            throw invalid_argument("[" + nome + "]: Dimensões do gradiente não correspondem à saída");
        }
        // gradiente em relação a ativação(aplica derivada)
        vector<vector<vector<float>>> gradAtivacao = gradienteSaida;
        
        for(size_t f = 0; f < filtros; f++) {
            for(size_t y = 0; y < saidaAltura; y++) {
                for(size_t x = 0; x < saidaLargura; x++) {
                    gradAtivacao[f][y][x] *= derivadaAtivacao(saidaCache[f][y][x]);
                }
            }
        }
        // calcula gradientes dos pesos
        calcularGradPesos(gradAtivacao);
        
        // calcula gradientes do bias
        if(usarBias) calcularGradBias(gradAtivacao);
        
        // calcula gradiente pra camada anterior(entrada)
        return calcularGradEntrada(gradAtivacao);
    }
    
    void calcularGradPesos(const vector<vector<vector<float>>>& gradAtivacao) {
        // pra cada filtro
        for(size_t f = 0; f < filtros; f++) {
            // pra cada canal
            for(size_t c = 0; c < canalEntrada; c++) {
                // pra cada posição do filtro
                for(size_t i = 0; i < alturaFiltro; i++) {
                    for(size_t j = 0; j < larguraFiltro; j++) {
                        float soma = 0.0f;
                        
                        // percorre todas as posições do gradiente
                        for(size_t y = 0; y < saidaAltura; y++) {
                            for(size_t x = 0; x < saidaLargura; x++) {
                                size_t entradaY = y * passo + i;
                                size_t entradaX = x * passo + j;
                                
                                soma += gradAtivacao[f][y][x] * entradaCache[c][entradaY][entradaX];
                            }
                        }
                        gradPesos[f][c][i][j] += soma;
                    }
                }
            }
        }
    }
    
    void calcularGradBias(const vector<vector<vector<float>>>& gradAtivacao) {
        for(size_t f = 0; f < filtros; f++) {
            float soma = 0.0f;
            
            for(size_t y = 0; y < saidaAltura; y++) {
                for(size_t x = 0; x < saidaLargura; x++) {
                    soma += gradAtivacao[f][y][x];
                }
            }
            gradBias[f] += soma;
        }
    }
    
    vector<vector<vector<float>>> calcularGradEntrada(const vector<vector<vector<float>>>& gradAtivacao) {
        // gradiente pra a entrada(com espaco)
        vector<vector<vector<float>>> gradentradaComEspaco = zeros3D(canalEntrada,
        entradaAltura + 2 * espaco, entradaLargura + 2 * espaco);
        
        // pra cada filtro
        for(size_t f = 0; f < filtros; f++) {
            // pra cada posição no gradiente de saida
            for(size_t y = 0; y < saidaAltura; y++) {
                for(size_t x = 0; x < saidaLargura; x++) {
                    float grad = gradAtivacao[f][y][x];
                    
                    // pra cada canal
                    for(size_t c = 0; c < canalEntrada; c++) {
                        // pra cada posição do filtro
                        for(size_t i = 0; i < alturaFiltro; i++) {
                            for(size_t j = 0; j < larguraFiltro; j++) {
                                size_t entradaY = y * passo + i;
                                size_t entradaX = x * passo + j;
                                
                                gradentradaComEspaco[c][entradaY][entradaX] += grad * pesos[f][c][i][j];
                            }
                        }
                    }
                }
            }
        }
        // remove espaco se necessario
        if(espaco == 0) return gradentradaComEspaco;
        
        vector<vector<vector<float>>> gradEntrada(canalEntrada, 
        vector<vector<float>>(entradaAltura, vector<float>(entradaLargura, 0.0f)));
        
        for(size_t c = 0; c < canalEntrada; c++) {
            for(size_t i = 0; i < entradaAltura; i++) {
                for(size_t j = 0; j < entradaLargura; j++) {
                    gradEntrada[c][i][j] = gradentradaComEspaco[c][i + espaco][j + espaco];
                }
            }
        }
        return gradEntrada;
    }
    
    void att(float taxaAprendizado) override {
        if(otimizador) {
            // prepara os pesos em formato 2D pro otimizador
            vector<vector<float>> pesos2D = converterPesos2D();
            vector<vector<float>> gradPesos2D = converterGradPesos2D();
            
            // converte bias pra matriz 2D(1xN)
            vector<vector<float>> bias2D = {bias};
            vector<vector<float>> gradBias2D = {gradBias};
            
            otimizador->att(pesos2D, gradPesos2D, bias, gradBias);
            
            // reconverte pesos de volta pra 4D
            reconverterPesos2D(pesos2D);
        } else {
            // atualiza SGD padrão
            for(size_t f = 0; f < filtros; f++) {
                for(size_t c = 0; c < canalEntrada; c++) {
                    for(size_t i = 0; i < alturaFiltro; i++) {
                        for(size_t j = 0; j < larguraFiltro; j++) {
                            pesos[f][c][i][j] -= taxaAprendizado * gradPesos[f][c][i][j];
                        }
                    }
                }
            }
            if(usarBias) {
                for(size_t f = 0; f < filtros; f++) {
                    bias[f] -= taxaAprendizado * gradBias[f];
                }
            }
        }
    }
    // converte pesos 4D pra 2D
    vector<vector<float>> converterPesos2D() const {
        size_t totalElementos = filtros * canalEntrada * alturaFiltro * larguraFiltro;
        vector<vector<float>> pesos2D(1, vector<float>(totalElementos));
        
        size_t idc = 0;
        for(size_t f = 0; f < filtros; f++) {
            for(size_t c = 0; c < canalEntrada; c++) {
                for(size_t i = 0; i < alturaFiltro; i++) {
                    for(size_t j = 0; j < larguraFiltro; j++) {
                        pesos2D[0][idc++] = pesos[f][c][i][j];
                    }
                }
            }
        }
        return pesos2D;
    }
    
    // converte gradientes de pesos 4D pra 2D
    vector<vector<float>> converterGradPesos2D() const {
        size_t totalElementos = filtros * canalEntrada * alturaFiltro * larguraFiltro;
        vector<vector<float>> grad2D(1, vector<float>(totalElementos));
        
        size_t idc = 0;
        for(size_t f = 0; f < filtros; f++) {
            for(size_t c = 0; c < canalEntrada; c++) {
                for(size_t i = 0; i < alturaFiltro; i++) {
                    for(size_t j = 0; j < larguraFiltro; j++) {
                        grad2D[0][idc++] = gradPesos[f][c][i][j];
                    }
                }
            }
        }
        return grad2D;
    }
    
    // reconverte pesos 2D pra 4D
    void reconverterPesos2D(const vector<vector<float>>& pesos2D) {
        size_t idc = 0;
        for(size_t f = 0; f < filtros; f++) {
            for(size_t c = 0; c < canalEntrada; c++) {
                for(size_t i = 0; i < alturaFiltro; i++) {
                    for(size_t j = 0; j < larguraFiltro; j++) {
                        pesos[f][c][i][j] = pesos2D[0][idc++];
                    }
                }
            }
        }
    }
    
    void zerarGradientes() override {
        // zera gradientes dos pesos
        for(auto& filtro : gradPesos) {
            for(auto& canal : filtro) {
                for(auto& linha : canal) {
                    fill(linha.begin(), linha.end(), 0.0f);
                }
            }
        }
        // zera gradientes do bias
        fill(gradBias.begin(), gradBias.end(), 0.0f);
    }
    
    bool temParametros() const override { return true; }
    size_t numParametros() const override {
        size_t pesosParams = filtros * canalEntrada * alturaFiltro * larguraFiltro;
        size_t biasParams = usarBias ? filtros : 0;
        return pesosParams + biasParams;
    }
    
    // serialização
    void salvar(const string& nomeArquivo) const override {
        ofstream arquivo(nomeArquivo);
        if(!arquivo) throw runtime_error("[" + nome + "]: Não foi possível salvar Conv2D");
        
        arquivo << "CONV2D_CAMADA" << endl;
        arquivo << filtros << " " << alturaFiltro << " " << larguraFiltro << " "
                << canalEntrada << " " << passo << " " << espaco << endl;
        arquivo << tipoAtivacao << " " << (usarBias ? 1 : 0) << endl;
        
        // salva pesos
        for(size_t f = 0; f < filtros; f++) {
            for(size_t c = 0; c < canalEntrada; c++) {
                for(size_t i = 0; i < alturaFiltro; i++) {
                    for(size_t j = 0; j < larguraFiltro; j++) {
                        arquivo << pesos[f][c][i][j] << " ";
                    }
                }
            }
        }
        arquivo << endl;
        
        // salva bias
        if(usarBias) {
            for(size_t f = 0; f < filtros; f++) {
                arquivo << bias[f] << " ";
            }
            arquivo << endl;
        }
        arquivo.close();
    }
    
    void carregar(const string& nomeArquivo) override {
        ifstream arquivo(nomeArquivo);
        if(!arquivo) throw runtime_error("[" + nome + "]: Não foi possível carregar Conv2D");
        
        string tipo;
        arquivo >> tipo;
        if(tipo != "CONV2D_CAMADA") {
            throw runtime_error("[" + nome + "]: Formato de arquivo inválido");
        }
        arquivo >> filtros >> alturaFiltro >> larguraFiltro 
        >> canalEntrada >> passo >> espaco;
        
        int usarBiasInt;
        arquivo >> tipoAtivacao >> usarBiasInt;
        usarBias = (usarBiasInt == 1);
        
        configAtivacao(tipoAtivacao);
        
        // redimensiona pesos
        pesos.resize(filtros);
        for(size_t f = 0; f < filtros; f++) {
            pesos[f].resize(canalEntrada);
            for(size_t c = 0; c < canalEntrada; c++) {
                pesos[f][c].resize(alturaFiltro);
                for(size_t i = 0; i < alturaFiltro; i++) {
                    pesos[f][c][i].resize(larguraFiltro);
                    for(size_t j = 0; j < larguraFiltro; j++) {
                        arquivo >> pesos[f][c][i][j];
                    }
                }
            }
        }
        // carrega bias
        if(usarBias) {
            bias.resize(filtros);
            for(size_t f = 0; f < filtros; f++) {
                arquivo >> bias[f];
            }
        }
        arquivo.close();
        
        // reinicializa gradientes
        iniciarGrad();
    }
};
// camada de reuso 2D
class MaxReuso2D : public Camada {
public:
    size_t tamReuso; // tamanho da janela de reuso
    size_t passo; // passo do reuso
    
    // cache pra retropropagação
    vector<vector<vector<vector<pair<size_t, size_t>>>>> indiceCache;  // [batch][canal][altura][largura] -> (y,x) do máximo
    vector<vector<vector<vector<float>>>> entradaCache;  // entrada original
    
    // dimensões
    size_t entradaCanais, entradaAltura, entradaLargura;
    size_t saidaAltura, saidaLargura;
    
    MaxReuso2D(size_t tamReuso = 2, size_t passo = 2,
    const string& nome = "")
    : Camada(nome), tamReuso(tamReuso), passo(passo) {
        tipo = "MaxReuso2D";
    }
    
    vector<float> prop(const vector<float>& entrada) override {
        throw runtime_error("[" + nome + "]: Use propMapa para MaxReuso2D");
    }
    
    vector<vector<vector<float>>> propMapa(const vector<vector<vector<float>>>& entrada) override {
        entradaCanais = entrada.size();
        entradaAltura = entrada[0].size();
        entradaLargura = entrada[0][0].size();
        
        // calcula dimensões de saida
        saidaAltura = (entradaAltura - tamReuso) / passo + 1;
        saidaLargura = (entradaLargura - tamReuso) / passo + 1;
        
        if(saidaAltura <= 0 || saidaLargura <= 0) {
            throw invalid_argument("[" + nome + "]: Dimensões de saída inválidas para pooling");
        }
        // armazena entrada no cache
        entradaCache = {entrada};
        
        // inicializa cache de indices
        indiceCache.resize(1);
        indiceCache[0].resize(entradaCanais);
        for(size_t c = 0; c < entradaCanais; c++) {
            indiceCache[0][c].resize(saidaAltura);
            for(size_t i = 0; i < saidaAltura; i++) {
                indiceCache[0][c][i].resize(saidaLargura);
            }
        }
        
        // cria tensor de saida
        vector<vector<vector<float>>> saida(entradaCanais);
        
        // pra cada canal
        for(size_t c = 0; c < entradaCanais; c++) {
            saida[c].resize(saidaAltura);
            
            for(size_t y = 0; y < saidaAltura; y++) {
                saida[c][y].resize(saidaLargura);
                
                for(size_t x = 0; x < saidaLargura; x++) {
                    // encontra maximo na região de reuso
                    float maxVal = -INFINITY;
                    size_t maxY = 0, maxX = 0;
                    
                    for(size_t i = 0; i < tamReuso; i++) {
                        for(size_t j = 0; j < tamReuso; j++) {
                            size_t entradaY = y * passo + i;
                            size_t entradaX = x * passo + j;
                            
                            if(entradaY < entradaAltura && entradaX < entradaLargura) {
                                float val = entrada[c][entradaY][entradaX];
                                if(val > maxVal) {
                                    maxVal = val;
                                    maxY = entradaY;
                                    maxX = entradaX;
                                }
                            }
                        }
                    }
                    saida[c][y][x] = maxVal;
                    
                    // armazena posição do maximo para retropropagação
                    indiceCache[0][c][y][x] = pair<size_t, size_t>(maxY, maxX);
                }
            }
        }
        return saida;
    }
    
    vector<vector<vector<vector<float>>>> propLoteMapa(const vector<vector<vector<vector<float>>>>& entradaLote) override {
        vector<vector<vector<vector<float>>>> saidaLote;
        
        for(const auto& entrada : entradaLote) {
            saidaLote.push_back(propMapa(entrada));
        }
        return saidaLote;
    }
    
    vector<float> retroprop(const vector<float>& gradiente) override {
        throw runtime_error("[" + nome + "]: Use retropropMapa para MaxReuso2D");
    }
    
    vector<vector<vector<float>>> retropropMapa(const vector<vector<vector<float>>>& gradienteSaida) override {
        if(gradienteSaida.size() != entradaCanais) {
            throw invalid_argument("[" + nome + "]: Dimensão do gradiente incorreta");
        }
        // gradiente pra entrada
        vector<vector<vector<float>>> gradEntrada(entradaCanais, 
        vector<vector<float>>(entradaAltura, 
        vector<float>(entradaLargura, 0.0f)));
        // pra cada canal
        for(size_t c = 0; c < entradaCanais; c++) {
            // Para cada posição na saída
            for(size_t y = 0; y < saidaAltura; y++) {
                for(size_t x = 0; x < saidaLargura; x++) {
                    // obtém posição do maximo
                    auto [maxY, maxX] = indiceCache[0][c][y][x];
                    
                    // propaga gradiente apenas pra posição do maximo
                    gradEntrada[c][maxY][maxX] += gradienteSaida[c][y][x];
                }
            }
        }
        return gradEntrada;
    }
    // maxPool não tem parametros pra atualizar
    void att(float taxaAprendizado) override {}
    // maxPool não tem gradientes
    void zerarGradientes() override {}
    
    bool temParametros() const override { return false; }
    size_t numParametros() const override { return 0; }
    
    void salvar(const string& nomeArquivo) const override {
        ofstream arquivo(nomeArquivo);
        if(!arquivo) throw runtime_error("[" + nome + "]: Não foi possível salvar MaxReuso2D");
        
        arquivo << "MaxReuso2D_CAMADA" << endl;
        arquivo << tamReuso << " " << passo << endl;
        
        arquivo.close();
    }
    
    void carregar(const string& nomeArquivo) override {
        ifstream arquivo(nomeArquivo);
        if(!arquivo) throw runtime_error("[" + nome + "]: Não foi possível carregar MaxReuso2D");
        
        string tipo;
        arquivo >> tipo;
        if(tipo != "MaxReuso2D_CAMADA") {
            throw runtime_error("[" + nome + "]: Formato de arquivo inválido");
        }
        arquivo >> tamReuso >> passo;
        
        arquivo.close();
    }
};
// camada de flatten(pra converter 3D/4D em 1D/2D)
class Flatten : public Camada {
public:
    size_t entradaCanais, entradaAltura, entradaLargura;
    size_t saidaDimensao;
    
    Flatten(const string& nome = "") : Camada(nome) {
        tipo = "Flatten";
    }
    vector<float> prop(const vector<float>& entrada) override {
        throw runtime_error("[" + nome + "]: Use propMapa para Flatten");
    }
    
    vector<vector<vector<float>>> propMapa(const vector<vector<vector<float>>>& entrada) override {
        entradaCanais = entrada.size();
        entradaAltura = entrada[0].size();
        entradaLargura = entrada[0][0].size();
        
        saidaDimensao = entradaCanais * entradaAltura * entradaLargura;
        
        // retorna em formato 3D: 1 canal x 1 linha x N colunas
        vector<vector<vector<float>>> saida(1);
        saida[0].resize(1);
        saida[0][0].resize(saidaDimensao);
        
        size_t idx = 0;
        for(size_t c = 0; c < entradaCanais; c++) {
            for(size_t i = 0; i < entradaAltura; i++) {
                for(size_t j = 0; j < entradaLargura; j++) {
                    saida[0][0][idx++] = entrada[c][i][j];
                }
            }
        }
        return saida;
    }
    
    vector<float> propMapa1D(const vector<vector<vector<float>>>& entrada) {
        auto saida3D = propMapa(entrada);
        return saida3D[0][0];
    }
    
    vector<vector<vector<vector<float>>>> propLoteMapa(const vector<vector<vector<vector<float>>>>& entradaLote) override {
        vector<vector<vector<vector<float>>>> saidaLote;
        
        for(const auto& entrada : entradaLote) {
            saidaLote.push_back(propMapa(entrada));
        }
        return saidaLote;
    }
    
    vector<vector<float>> propLoteMapa2D(const vector<vector<vector<vector<float>>>>& entradaLote) {
        vector<vector<float>> saidaLote2D;
        
        for(const auto& entrada : entradaLote) {
            saidaLote2D.push_back(propMapa1D(entrada));
        }
        return saidaLote2D;
    }
    
    vector<float> retroprop(const vector<float>& gradiente) override {
        throw runtime_error("[" + nome + "]: Use retropropMapa para Flatten");
    }
    vector<vector<vector<float>>> retropropMapa(const vector<vector<vector<float>>>& gradienteSaida) override {
        // o gradiente de saída chega em formato 3D: 1 canal x 1 linha x N colunas
        if(gradienteSaida.size() != 1 || gradienteSaida[0].size() != 1) {
            throw invalid_argument("[" + nome + "]: Formato do gradiente incorreto para Flatten");
        }
        const vector<float>& gradiente1D = gradienteSaida[0][0];
        
        if(gradiente1D.size() != saidaDimensao) {
            throw invalid_argument("[" + nome + "]: Dimensão do gradiente incorreta");
        }
        vector<vector<vector<float>>> gradEntrada(entradaCanais, 
            vector<vector<float>>(entradaAltura, 
            vector<float>(entradaLargura, 0.0f)));
        
        size_t idc = 0;
        for(size_t c = 0; c < entradaCanais; c++) {
            for(size_t i = 0; i < entradaAltura; i++) {
                for(size_t j = 0; j < entradaLargura; j++) {
                    gradEntrada[c][i][j] = gradiente1D[idc++];
                }
            }
        }
        return gradEntrada;
    }
    
    vector<vector<vector<float>>> retropropMapa1D(const vector<float>& gradiente1D) {
        vector<vector<vector<float>>> gradiente3D(1);
        gradiente3D[0].resize(1);
        gradiente3D[0][0] = gradiente1D;
        return retropropMapa(gradiente3D);
    }
    // flatten não tem parametros
    void att(float taxaAprendizado) override {}
    // flatten não tem gradientes
    void zerarGradientes() override {}
    
    bool temParametros() const override { return false; }
    size_t numParametros() const override { return 0; }
    
    void salvar(const string& nomeArquivo) const override {
        ofstream arquivo(nomeArquivo);
        if(!arquivo) throw runtime_error("[" + nome + "]: Não foi possível salvar Flatten");
        
        arquivo << "FLATTEN_CAMADA" << endl;
        
        arquivo.close();
    }
    
    void carregar(const string& nomeArquivo) override {
        ifstream arquivo(nomeArquivo);
        if(!arquivo) throw runtime_error("[" + nome + "]: Não foi possível carregar Flatten");
        
        string tipo;
        arquivo >> tipo;
        if(tipo != "FLATTEN_CAMADA") {
            throw runtime_error("[" + nome + "]: Formato de arquivo inválido");
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
    function<float(const vector<float>&, const vector<float>&)> perda = mse,
    function<vector<float>(const vector<float>&, const vector<float>&)> derivadaPerda = derivadaMse,
    float taxaAprendizado = 0.01f) {
        // garante que ta no modo treino
        modoTreino();
        
        // propagação
        vector<float> saida = prop(entrada);
        
        // calcula a perda
        float erro = perda(saida, alvo);
        
        // gradiente inicial
        vector<float> gradiente = derivadaPerda(saida, alvo);
        
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
    
    // pra CNNs:
    float treinarMapa(const vector<vector<vector<float>>>& entrada3D, 
    const vector<float>& alvo,
    function<float(const vector<float>&, const vector<float>&)> perda = mse,
    function<vector<float>(const vector<float>&, const vector<float>&)> derivadaPerda = derivadaMse,
    float taxaAprendizado = 0.01f) {
        modoTreino();
        zerarGradientes();
        
        // propagação pra CNN
        vector<vector<vector<float>>> resultado3D = entrada3D;
        
        // propaga atraves de todas as camadas
        for(size_t i = 0; i < camadas.size(); i++) {
            auto* conv2d = dynamic_cast<Conv2D*>(camadas[i].get());
            auto* maxpool = dynamic_cast<MaxReuso2D*>(camadas[i].get());
            auto* flatten = dynamic_cast<Flatten*>(camadas[i].get());
            auto* densa = dynamic_cast<Densa*>(camadas[i].get());
            
            if(conv2d) resultado3D = conv2d->propMapa(resultado3D);
            else if(maxpool) resultado3D = maxpool->propMapa(resultado3D);
            else if(flatten) resultado3D = flatten->propMapa(resultado3D);
            else if(densa) {
                // verifica dimensão antes de processar
                vector<float> entrada1D = resultado3D[0][0];
                if(entrada1D.size() != densa->entradaDim) {
                    throw runtime_error("[" + densa->nome + "]: Dimensão de entrada incorreta. Recebido: " + 
                    to_string(entrada1D.size()) + ", Esperado: " + to_string(densa->entradaDim));
                }
                auto saida1D = densa->prop(entrada1D);
                
                // coloca de volta no formato 3D pra consistencia
                resultado3D = vector<vector<vector<float>>>(1, vector<vector<float>>(1, saida1D));
            }
        }
        // extrai resultado final
        vector<float> saidaFinal = resultado3D[0][0];
        
        // calcula perda
        float erro = perda(saidaFinal, alvo);
        
        // calcula gradiente inicial
        vector<float> gradiente1D = derivadaPerda(saidaFinal, alvo);
        // converte gradiente 1D pra 3D pra retropropagação
        vector<vector<vector<float>>> gradiente3D(1, vector<vector<float>>(1, gradiente1D));
        
        // retropropagação na ordem inversa
        for(int i = camadas.size() - 1; i >= 0; i--) {
            auto* conv2d = dynamic_cast<Conv2D*>(camadas[i].get());
            auto* maxpool = dynamic_cast<MaxReuso2D*>(camadas[i].get());
            auto* flatten = dynamic_cast<Flatten*>(camadas[i].get());
            auto* densa = dynamic_cast<Densa*>(camadas[i].get());
            
            if(densa) {
                // densa: gradiente ja ta em formato 3D(1x1xN)
                gradiente3D = densa->retropropMapa(gradiente3D);
            } else if(flatten) {
                // flatten: gradiente chega em formato 3D, retropropaga pra 3D
                gradiente3D = flatten->retropropMapa(gradiente3D);
            } else if(maxpool) {
                // maxPool: retropropaga em formato 3D
                gradiente3D = maxpool->retropropMapa(gradiente3D);
            } else if(conv2d) {
                // conv2D: retropropaga em formato 3D
                gradiente3D = conv2d->retropropMapa(gradiente3D);
            }
        }
        // atualiza pesos
        att(taxaAprendizado);
        
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