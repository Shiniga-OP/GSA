// biblis/camadas/camada.h
#pragma once
#include "../ativas.h"
#include "../util.h"
#include "../otimis/otimizador.h"
#include <math.h>
#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <stdexcept>
#include <memory>

using namespace std;

// gradiente generico camadas simples usam apenas "vetor"
// camadas especiais podem usar "matriz"(2D) ou "cubo"(3D)
struct GradGenerico {
    vector<float> vetor; // gradiente 1D(dim,)
    vector<vector<float>> matriz; // gradiente 2D(m x dim)
    vector<vector<vector<float>>> cubo; // gradiente 3D(c x h x w)

    // aliases semanticos pra atenção
    vector<float>& gradEstado = vetor;
    vector<vector<float>>& gradChaves = matriz;

    GradGenerico() {}
    GradGenerico(vector<float> v) : vetor(std::move(v)) {}
    GradGenerico(vector<float> v, vector<vector<float>> m)
    : vetor(std::move(v)), matriz(std::move(m)) {}
    GradGenerico(vector<vector<vector<float>>> c) : cubo(std::move(c)) {}
};

class Camada {
public:
    string tipo;
    string nome;
    unique_ptr<Otimizador> otimizador;
    
    Camada(const string& nome = "") : nome(nome) {}
    virtual ~Camada() = default;
    
    virtual vector<float> prop(const vector<float>& entrada) = 0; // propagação
    virtual GradGenerico retroprop(const vector<float>& gradiente) = 0; // retropropagação

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
            res.push_back(retroprop(g).vetor);
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