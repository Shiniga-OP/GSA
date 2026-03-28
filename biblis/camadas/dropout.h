// biblis/camadas/dropout.h
#pragma once

#include "camada.h"

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
    
    GradGenerico retroprop(const vector<float>& gradiente) override {
        if(!treinando || taxa == 0.0f) {
            return GradGenerico(gradiente);
        }
        if(mascara.size() != gradiente.size()) {
            throw std::runtime_error("[" + nome + "]: Máscara não gerada na propagação");
        }
        vector<float> gradEntrada(gradiente.size());
        for(size_t i = 0; i < gradiente.size(); i++) {
            gradEntrada[i] = mascara[i] ? gradiente[i] : 0.0f;
        }
        return GradGenerico(gradEntrada);
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