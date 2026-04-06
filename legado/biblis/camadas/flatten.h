// biblis/camadas/flatten.h
#pragma once

#include "camada.h"

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
    
    GradGenerico retroprop(const vector<float>& gradiente) override {
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