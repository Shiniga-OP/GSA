// biblis/camadas/atencao.h
#pragma once
#include "camada.h"

// atenção de consulta unica sobre um conjunto de chaves/valores
// custo O(m) onde m = numero de entradas na memoria

// fluxo:
// estado(dim) -> Pq -> q(dimAtencao)
// chaves[i](dim) -> Pk -> k[i](dimAtencao)
// chaves[i](dim) -> Pv -> v[i](dimSaida)
// pesos = softmax(q * k[i] / sqrt(dimAtencao))
// saida = soma(pesos[i] * v[i])

// Pq, Pk, Pv são treinaveis por retropropagação

class CamadaAtencao : public Camada {
public:
    size_t dim; // dimensão da entrada(estado e chaves)
    size_t dimAtencao; // dimensão interna Q/K
    size_t dimSaida; // dimensão da saida(projeção V)

    // projeções treinaveis
    vector<vector<float>> Pq; // [dimAtencao x dim]
    vector<vector<float>> Pk; // [dimAtencao x dim]
    vector<vector<float>> Pv; // [dimSaida   x dim]

    // gradientes
    vector<vector<float>> gradPq;
    vector<vector<float>> gradPk;
    vector<vector<float>> gradPv;

    // otimizadores independentes por projeção
    unique_ptr<Otimizador> otimQ;
    unique_ptr<Otimizador> otimK;
    unique_ptr<Otimizador> otimV;

    // cache pra retropropagação
    vector<float> consultaCache;
    vector<vector<float>> chavesCache;// k[i] projetadas
    vector<vector<float>> valoresCache; // v[i] projetados
    vector<float> pesosCache; // softmax(pontos)
    vector<float> pontosCache; // q*k[i]/sqrt(d) antes do softmax
    vector<float> entradaCache; // estado original
    vector<vector<float>> chavesEntradaCache; // chaves originais(antes de Pk)

    float escala; // 1/sqrt(dimAtencao)

    CamadaAtencao(size_t dim, size_t dimAtencao, size_t dimSaida,
    const string& nome = "atencao")
        : Camada(nome),
          dim(dim), dimAtencao(dimAtencao), dimSaida(dimSaida),
          escala(1.0f / sqrt((float)dimAtencao)) {

        tipo = "CamadaAtencao";

        Pq = iniPesosXavier(dimAtencao, dim);
        Pk = iniPesosXavier(dimAtencao, dim);
        Pv = iniPesosXavier(dimSaida, dim);

        gradPq = matrizZeros(dimAtencao, dim);
        gradPk = matrizZeros(dimAtencao, dim);
        gradPv = matrizZeros(dimSaida, dim);
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

    vector<float> prop(const vector<float>& entrada) override {
        throw runtime_error("[" + nome + "]: use prop(estado, chaves)");
    }

    // prop real: estado(dim,) + chaves[](m x dim) -> saida(dimSaida,)
    vector<float> prop(
        const vector<float>& estado,
        const vector<vector<float>>& chaves
    ) {
        if(estado.size() != dim)
            throw invalid_argument("[" + nome + "]: dimensão do estado incorreta");
        if(chaves.empty())
            throw invalid_argument("[" + nome + "]: conjunto de chaves vazio");
        for(const auto& c : chaves) {
            if(c.size() != dim)
                throw invalid_argument("[" + nome + "]: dimensão de chave incorreta");
        }
        size_t m = chaves.size();

        entradaCache = estado;
        chavesEntradaCache = chaves;

        // q = Pq * estado
        consultaCache = aplicarMatriz(Pq, estado);

        // k[i] = Pk * chaves[i], v[i] = Pv * chaves[i]
        chavesCache.resize(m);
        valoresCache.resize(m);
        for(size_t i = 0; i < m; i++) {
            chavesCache[i] = aplicarMatriz(Pk, chaves[i]);
            valoresCache[i] = aplicarMatriz(Pv, chaves[i]);
        }
        // pontos = q * k[i] / sqrt(dimAtencao)
        pontosCache.resize(m);
        for(size_t i = 0; i < m; i++) {
            float dot = 0.0f;
            for(size_t j = 0; j < dimAtencao; j++) dot += consultaCache[j] * chavesCache[i][j];
            pontosCache[i] = dot * escala;
        }
        // softmax
        pesosCache = softmax(pontosCache);

        // saida = soma(pesos[i] * v[i])
        vector<float> saida(dimSaida, 0.0f);
        for(size_t i = 0; i < m; i++) {
            for(size_t j = 0; j < dimSaida; j++) {
                saida[j] += pesosCache[i] * valoresCache[i][j];
            }
        }
        return saida;
    }

    // retroprop: gradSaida(dimSaida,) ->
    // acumula gradPq, gradPk, gradPv
    // retorna GradGenerico com:
    // vetor = gradEstado (dim,)
    // matriz = gradChaves (m x dim)
    GradGenerico retroprop(const vector<float>& gradSaida) override {
        if(gradSaida.size() != dimSaida) throw invalid_argument("[" + nome + "]: dimensão do gradiente de saída incorreta");

        size_t m = pesosCache.size();

        // dL/dv[i] = pesos[i] * gradSaida
        vector<vector<float>> gradV(m, vector<float>(dimSaida));
        for(size_t i = 0; i < m; i++) {
            for(size_t j = 0; j < dimSaida; j++) {
                gradV[i][j] = pesosCache[i] * gradSaida[j];
            }
        }
        // dL/dpesos[i] = gradSaida · v[i]
        vector<float> gradPesos(m);
        for(size_t i = 0; i < m; i++) {
            float dot = 0.0f;
            for(size_t j = 0; j < dimSaida; j++) dot += gradSaida[j] * valoresCache[i][j];
            gradPesos[i] = dot;
        }
        // gradiente pelo softmax
        vector<float> gradpontos(m);
        float soma = 0.0f;
        for(size_t i = 0; i < m; i++) soma += gradPesos[i] * pesosCache[i];
        for(size_t i = 0; i < m; i++) gradpontos[i] = pesosCache[i] * (gradPesos[i] - soma);

        // escala
        for(size_t i = 0; i < m; i++) gradpontos[i] *= escala;

        // dL/dq = soma(gradpontos[i] * k[i])
        vector<float> gradQ(dimAtencao, 0.0f);
        for(size_t i = 0; i < m; i++) {
            for(size_t j = 0; j < dimAtencao; j++) {
                gradQ[j] += gradpontos[i] * chavesCache[i][j];
            }
        }
        // dL/dk[i] = gradpontos[i] * q
        vector<vector<float>> gradK(m, vector<float>(dimAtencao));
        for(size_t i = 0; i < m; i++) {
            for(size_t j = 0; j < dimAtencao; j++) {
                gradK[i][j] = gradpontos[i] * consultaCache[j];
            }
        }
        // acumula gradPq: dL/dPq = gradQ ⊗ estado
        for(size_t i = 0; i < dimAtencao; i++) {
            for(size_t j = 0; j < dim; j++) {
                gradPq[i][j] += gradQ[i] * entradaCache[j];
            }
        }
        // acumula gradPk e gradPv
        for(size_t i = 0; i < m; i++) {
            for(size_t a = 0; a < dimAtencao; a++) {
                for(size_t b = 0; b < dim; b++) {
                    gradPk[a][b] += gradK[i][a] * chavesEntradaCache[i][b];
                }
            }
            for(size_t a = 0; a < dimSaida; a++) {
                for(size_t b = 0; b < dim; b++) {
                    gradPv[a][b] += gradV[i][a] * chavesEntradaCache[i][b];
                }
            }
        }
        // dL/destado = Pq^T * gradQ
        vector<float> gradEstado(dim, 0.0f);
        for(size_t j = 0; j < dim; j++) {
            for(size_t i = 0; i < dimAtencao; i++) {
                gradEstado[j] += Pq[i][j] * gradQ[i];
            }
        }
        // dL/dchaves[i] = Pk^T * gradK[i] + Pv^T * gradV[i]
        vector<vector<float>> gradChaves(m, vector<float>(dim, 0.0f));
        for(size_t i = 0; i < m; i++) {
            for(size_t j = 0; j < dim; j++) {
                for(size_t a = 0; a < dimAtencao; a++) {
                    gradChaves[i][j] += Pk[a][j] * gradK[i][a];
                }
                for(size_t a = 0; a < dimSaida; a++) {
                    gradChaves[i][j] += Pv[a][j] * gradV[i][a];
                }
            }
        }
        return GradGenerico(gradEstado, gradChaves);
    }

    void att(float taxaAprendizado) override {
        vector<float> biasZero(1, 0.0f);
        vector<float> gradBiasZero(1, 0.0f);

        if(otimQ) otimQ->att(Pq, gradPq, biasZero, gradBiasZero);
        else {
            for(size_t i = 0; i < dimAtencao; i++)
                for(size_t j = 0; j < dim; j++)
                    Pq[i][j] -= taxaAprendizado * gradPq[i][j];
        }
        if(otimK) otimK->att(Pk, gradPk, biasZero, gradBiasZero);
        else {
            for(size_t i = 0; i < dimAtencao; i++)
                for(size_t j = 0; j < dim; j++)
                    Pk[i][j] -= taxaAprendizado * gradPk[i][j];
        }
        if(otimV) otimV->att(Pv, gradPv, biasZero, gradBiasZero);
        else {
            for(size_t i = 0; i < dimSaida; i++) {
                for(size_t j = 0; j < dim; j++) {
                    Pv[i][j] -= taxaAprendizado * gradPv[i][j];
                }
            }
        }
    }

    void zerarGradientes() override {
        for(auto& l : gradPq) fill(l.begin(), l.end(), 0.0f);
        for(auto& l : gradPk) fill(l.begin(), l.end(), 0.0f);
        for(auto& l : gradPv) fill(l.begin(), l.end(), 0.0f);
    }

    bool temParametros() const override { return true; }

    size_t numParametros() const override {
        return dimAtencao * dim + dimAtencao * dim + dimSaida * dim;
    }

    // retorna os pesos de atenção da ultima chamada(util pro sistema de memoria)
    const vector<float>& pesosAtencao() const { return pesosCache; }

    void salvar(const string& arquivo) const override {
        ofstream a(arquivo, ios::binary);
        if(!a) throw runtime_error("[" + nome + "]: falha ao abrir arquivo para salvar");

        auto salvarMatriz = [&](const vector<vector<float>>& M) {
            size_t linhas = M.size();
            size_t colunas = linhas > 0 ? M[0].size() : 0;
            a.write(reinterpret_cast<const char*>(&linhas), sizeof(linhas));
            a.write(reinterpret_cast<const char*>(&colunas), sizeof(colunas));
            for(const auto& linha : M)
                a.write(reinterpret_cast<const char*>(linha.data()), colunas * sizeof(float));
        };
        salvarMatriz(Pq);
        salvarMatriz(Pk);
        salvarMatriz(Pv);
    }

    void carregar(const string& arquivo) override {
        ifstream a(arquivo, ios::binary);
        if(!a) throw runtime_error("[" + nome + "]: falha ao abrir arquivo para carregar");

        auto carregarMatriz = [&](vector<vector<float>>& M) {
            size_t linhas, colunas;
            a.read(reinterpret_cast<char*>(&linhas), sizeof(linhas));
            a.read(reinterpret_cast<char*>(&colunas), sizeof(colunas));
            M.assign(linhas, vector<float>(colunas));
            for(auto& linha : M)
                a.read(reinterpret_cast<char*>(linha.data()), colunas * sizeof(float));
        };
        carregarMatriz(Pq);
        carregarMatriz(Pk);
        carregarMatriz(Pv);
    }
};