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
    virtual vector<vector<float>> propLote(const vector<vector<float>>& entrada) {
        // processa cada exemplo sozinho
        vector<vector<float>> saida;
        for(const auto& e : entrada) {
            saida.push_back(prop(e));
        }
        return saida;
    }
    virtual vector<float> retroprop(const vector<float>& gradiente) = 0; // retropropagação
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
// modelo pra gerenciamento de camadas:
class Modelo {
public:
    vector<unique_ptr<Camada>> camadas;
    string nome;
    bool treinando;
    
    Modelo(const string& nome = "") : nome(nome), treinando(true) {}
    
    void adicionar(unique_ptr<Camada> camada) {
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
    
    void modoTreino() {
        treinando = true;
        for(auto& camada : camadas) {
            // pra dropout, chama metodo especifico
            auto* dropout = dynamic_cast<Dropout*>(camada.get());
            if(dropout) dropout->treinando = true;
        }
    }
    
    void modoTeste() {
        treinando = false;
        for(auto& camada : camadas) {
            auto* dropout = dynamic_cast<Dropout*>(camada.get());
            if(dropout) {
                dropout->treinando = false;
            }
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