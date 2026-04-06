// biblis/camadas/densa.h
#pragma once
#include "camada.h"

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
    
    GradGenerico retroprop(const vector<float>& gradiente) override {
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
        return GradGenerico(gradEntrada);
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
        auto gradEntrada1D = retroprop(gradiente1D).vetor;

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