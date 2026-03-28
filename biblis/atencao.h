// biblis/atencao.h
#pragma once
#include <vector>
#include <cmath>
#include <stdexcept>
#include <memory>
#include "util.h"
#include "otimizadores.h"

using namespace std;

// atenção de query única sobre um conjunto de chaves/valores
// custo O(m) onde m = número de entradas na memória
//
// fluxo:
//   estado (dim)  →  Wq  →  q (dimAtencao)
//   chaves[i] (dim) →  Wk  →  k[i] (dimAtencao)
//   chaves[i] (dim) →  Wv  →  v[i] (dimSaida)
//   pesos = softmax(q · k[i] / sqrt(dimAtencao))
//   saida = soma(pesos[i] * v[i])
//
// Wq, Wk, Wv são treináveis por retropropagação

class CamadaAtencao {
public:
    size_t dim;        // dimensão da entrada (estado e chaves)
    size_t dimAtencao; // dimensão interna Q/K
    size_t dimSaida;   // dimensão da saída (projeção V)

    // projeções treináveis
    vector<vector<float>> Wq; // [dimAtencao x dim]
    vector<vector<float>> Wk; // [dimAtencao x dim]
    vector<vector<float>> Wv; // [dimSaida   x dim]

    // gradientes
    vector<vector<float>> gradWq;
    vector<vector<float>> gradWk;
    vector<vector<float>> gradWv;

    // otimizadores independentes por projeção
    unique_ptr<Otimizador> otimQ;
    unique_ptr<Otimizador> otimK;
    unique_ptr<Otimizador> otimV;

    // cache pra retropropagação
    vector<float> queryCache;           // q projetada
    vector<vector<float>> chavesCache;  // k[i] projetadas
    vector<vector<float>> valoresCache; // v[i] projetados
    vector<float> pesosCache;           // softmax(scores)
    vector<float> scoresCache;          // q·k[i]/sqrt(d) antes do softmax
    vector<float> entradaCache;         // estado original
    vector<vector<float>> chavesEntradaCache; // chaves originais (antes de Wk)

    float escala; // 1/sqrt(dimAtencao)

    CamadaAtencao(size_t dim, size_t dimAtencao, size_t dimSaida)
        : dim(dim), dimAtencao(dimAtencao), dimSaida(dimSaida),
          escala(1.0f / sqrt((float)dimAtencao)) {

        // Xavier pra Q e K (linear), He pra V (pode ter relu depois)
        Wq = iniPesosXavier(dimAtencao, dim);
        Wk = iniPesosXavier(dimAtencao, dim);
        Wv = iniPesosXavier(dimSaida, dim);

        gradWq = matrizZeros(dimAtencao, dim);
        gradWk = matrizZeros(dimAtencao, dim);
        gradWv = matrizZeros(dimSaida, dim);
    }

    void defOtimizadores(
        unique_ptr<Otimizador> oQ,
        unique_ptr<Otimizador> oK,
        unique_ptr<Otimizador> oV
    ) {
        otimQ = std::move(oQ);
        otimK = std::move(oK);
        otimV = std::move(oV);
    }

    // prop: estado (dim,) + chaves[] (m x dim) → saida (dimSaida,)
    // chaves são tanto as chaves de busca quanto os vetores de valor (mesmo espaço de entrada)
    // Wk e Wv projetam pra espaços diferentes
    vector<float> prop(
        const vector<float>& estado,
        const vector<vector<float>>& chaves
    ) {
        if(estado.size() != dim)
            throw invalid_argument("[CamadaAtencao]: dimensão do estado incorreta");
        if(chaves.empty())
            throw invalid_argument("[CamadaAtencao]: conjunto de chaves vazio");
        for(const auto& c : chaves)
            if(c.size() != dim)
                throw invalid_argument("[CamadaAtencao]: dimensão de chave incorreta");

        size_t m = chaves.size();

        // salva entradas no cache
        entradaCache = estado;
        chavesEntradaCache = chaves;

        // projeta query: q = Wq * estado
        queryCache = aplicarMatriz(Wq, estado);

        // projeta chaves e valores
        chavesCache.resize(m);
        valoresCache.resize(m);
        for(size_t i = 0; i < m; i++) {
            chavesCache[i]  = aplicarMatriz(Wk, chaves[i]);
            valoresCache[i] = aplicarMatriz(Wv, chaves[i]);
        }

        // calcula scores: q · k[i] / sqrt(dimAtencao)
        scoresCache.resize(m);
        for(size_t i = 0; i < m; i++) {
            float dot = 0.0f;
            for(size_t j = 0; j < dimAtencao; j++)
                dot += queryCache[j] * chavesCache[i][j];
            scoresCache[i] = dot * escala;
        }

        // softmax dos scores
        pesosCache = softmax(scoresCache);

        // saída ponderada: soma(pesos[i] * v[i])
        vector<float> saida(dimSaida, 0.0f);
        for(size_t i = 0; i < m; i++)
            for(size_t j = 0; j < dimSaida; j++)
                saida[j] += pesosCache[i] * valoresCache[i][j];

        return saida;
    }

    // retroprop: gradiente da saída (dimSaida,) →
    //   acumula gradWq, gradWk, gradWv
    //   retorna gradiente pro estado de entrada (dim,)
    // gradientes pras chaves também são retornados (m x dim)
    struct GradAtencao {
        vector<float> gradEstado;          // (dim,)
        vector<vector<float>> gradChaves;  // (m x dim)
    };

    GradAtencao retroprop(const vector<float>& gradSaida) {
        if(gradSaida.size() != dimSaida)
            throw invalid_argument("[CamadaAtencao]: dimensão do gradiente de saída incorreta");

        size_t m = pesosCache.size();

        // === gradiente em relação aos valores projetados v[i] ===
        // saida = sum(pesos[i] * v[i])
        // dL/dv[i] = pesos[i] * gradSaida
        vector<vector<float>> gradV(m, vector<float>(dimSaida));
        for(size_t i = 0; i < m; i++)
            for(size_t j = 0; j < dimSaida; j++)
                gradV[i][j] = pesosCache[i] * gradSaida[j];

        // === gradiente em relação aos pesos de atenção ===
        // saida = sum(pesos[i] * v[i])
        // dL/dpesos[i] = gradSaida · v[i]
        vector<float> gradPesos(m);
        for(size_t i = 0; i < m; i++) {
            float dot = 0.0f;
            for(size_t j = 0; j < dimSaida; j++)
                dot += gradSaida[j] * valoresCache[i][j];
            gradPesos[i] = dot;
        }

        // === gradiente através do softmax ===
        // gradScores = softmax_backward(pesos, gradPesos)
        // d softmax(s)[i]/ds[j] = pesos[i]*(delta_ij - pesos[j])
        vector<float> gradScores(m);
        float soma = 0.0f;
        for(size_t i = 0; i < m; i++) soma += gradPesos[i] * pesosCache[i];
        for(size_t i = 0; i < m; i++)
            gradScores[i] = pesosCache[i] * (gradPesos[i] - soma);

        // escala: scores = dot * escala → gradDot = gradScores * escala
        for(size_t i = 0; i < m; i++) gradScores[i] *= escala;

        // === gradiente em relação à query projetada q ===
        // score[i] = q · k[i]  →  dL/dq = sum(gradScores[i] * k[i])
        vector<float> gradQ(dimAtencao, 0.0f);
        for(size_t i = 0; i < m; i++)
            for(size_t j = 0; j < dimAtencao; j++)
                gradQ[j] += gradScores[i] * chavesCache[i][j];

        // === gradiente em relação às chaves projetadas k[i] ===
        // score[i] = q · k[i]  →  dL/dk[i] = gradScores[i] * q
        vector<vector<float>> gradK(m, vector<float>(dimAtencao));
        for(size_t i = 0; i < m; i++)
            for(size_t j = 0; j < dimAtencao; j++)
                gradK[i][j] = gradScores[i] * queryCache[j];

        // === acumula gradientes de Wq ===
        // q = Wq * estado  →  dL/dWq = gradQ ⊗ estado
        for(size_t i = 0; i < dimAtencao; i++)
            for(size_t j = 0; j < dim; j++)
                gradWq[i][j] += gradQ[i] * entradaCache[j];

        // === acumula gradientes de Wk e Wv ===
        // k[i] = Wk * chaves[i]  →  dL/dWk += gradK[i] ⊗ chaves[i]
        // v[i] = Wv * chaves[i]  →  dL/dWv += gradV[i] ⊗ chaves[i]
        for(size_t i = 0; i < m; i++) {
            for(size_t a = 0; a < dimAtencao; a++)
                for(size_t b = 0; b < dim; b++)
                    gradWk[a][b] += gradK[i][a] * chavesEntradaCache[i][b];

            for(size_t a = 0; a < dimSaida; a++)
                for(size_t b = 0; b < dim; b++)
                    gradWv[a][b] += gradV[i][a] * chavesEntradaCache[i][b];
        }

        // === gradiente pro estado de entrada ===
        // q = Wq * estado  →  dL/destado = Wq^T * gradQ
        vector<float> gradEstado(dim, 0.0f);
        for(size_t j = 0; j < dim; j++)
            for(size_t i = 0; i < dimAtencao; i++)
                gradEstado[j] += Wq[i][j] * gradQ[i];

        // === gradiente pras chaves de entrada ===
        // k[i] = Wk * chaves[i]  →  dL/dchaves[i] = Wk^T * gradK[i]
        // v[i] = Wv * chaves[i]  →  dL/dchaves[i] += Wv^T * gradV[i]
        vector<vector<float>> gradChaves(m, vector<float>(dim, 0.0f));
        for(size_t i = 0; i < m; i++) {
            for(size_t j = 0; j < dim; j++) {
                for(size_t a = 0; a < dimAtencao; a++)
                    gradChaves[i][j] += Wk[a][j] * gradK[i][a];
                for(size_t a = 0; a < dimSaida; a++)
                    gradChaves[i][j] += Wv[a][j] * gradV[i][a];
            }
        }

        return {gradEstado, gradChaves};
    }

    void att(float taxaAprendizado) {
        // bias dummy (atencao não tem bias — passa zeros)
        vector<float> biasZero(1, 0.0f);
        vector<float> gradBiasZero(1, 0.0f);

        if(otimQ) otimQ->att(Wq, gradWq, biasZero, gradBiasZero);
        else for(size_t i=0;i<dimAtencao;i++) for(size_t j=0;j<dim;j++) Wq[i][j] -= taxaAprendizado * gradWq[i][j];

        if(otimK) otimK->att(Wk, gradWk, biasZero, gradBiasZero);
        else for(size_t i=0;i<dimAtencao;i++) for(size_t j=0;j<dim;j++) Wk[i][j] -= taxaAprendizado * gradWk[i][j];

        if(otimV) otimV->att(Wv, gradWv, biasZero, gradBiasZero);
        else for(size_t i=0;i<dimSaida;i++) for(size_t j=0;j<dim;j++) Wv[i][j] -= taxaAprendizado * gradWv[i][j];
    }

    void zerarGradientes() {
        for(auto& l : gradWq) fill(l.begin(), l.end(), 0.0f);
        for(auto& l : gradWk) fill(l.begin(), l.end(), 0.0f);
        for(auto& l : gradWv) fill(l.begin(), l.end(), 0.0f);
    }

    size_t numParametros() const {
        return dimAtencao*dim + dimAtencao*dim + dimSaida*dim;
    }

    // retorna os pesos de atenção da última chamada (útil pro sistema de memória)
    const vector<float>& pesosAtencao() const { return pesosCache; }
};