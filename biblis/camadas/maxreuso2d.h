// biblis/camadas/maxreuso2d.h
#pragma once

#include "camada.h"

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
    
    GradGenerico retroprop(const vector<float>& gradiente) override {
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