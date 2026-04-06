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

// propLote: processa [T x dim] com máscara causal embutida
// cada token t atende apenas aos tokens 0..t

class CamadaAtencao : public Camada {
public:
    size_t dim;
    size_t dimAtencao;
    size_t dimSaida;

    vector<vector<float>> Pq; // [dimAtencao x dim]
    vector<vector<float>> Pk; // [dimAtencao x dim]
    vector<vector<float>> Pv; // [dimSaida x dim]

    vector<vector<float>> gradPq;
    vector<vector<float>> gradPk;
    vector<vector<float>> gradPv;

    unique_ptr<Otimizador> otimQ;
    unique_ptr<Otimizador> otimK;
    unique_ptr<Otimizador> otimV;

    // cache token a token(geração)
    vector<float> consultaCache;
    vector<vector<float>> chavesCache;
    vector<vector<float>> valoresCache;
    vector<float> pesosCache;
    vector<float> pontosCache;
    vector<float> entradaCache;
    vector<vector<float>> chavesEntradaCache;

    // cache lote(treino)
    struct CacheLote {
        vector<vector<float>> Q; // [T x dimAtencao]
        vector<vector<float>> K; // [T x dimAtencao]
        vector<vector<float>> V; // [T x dimSaida]
        vector<vector<float>> pesos; // [T x T](mascara causal aplicada)
        vector<vector<float>> entrada; // [T x dim]
    } cacheLote;

    float escala;

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

    vector<float> prop(
        const vector<float>& estado,
        const vector<vector<float>>& chaves
    ) {
        if(estado.size() != dim)
            throw invalid_argument("[" + nome + "]: dimensão do estado incorreta");
        if(chaves.empty())
            throw invalid_argument("[" + nome + "]: conjunto de chaves vazio");

        size_t m = chaves.size();
        entradaCache = estado;
        chavesEntradaCache = chaves;

        consultaCache = aplicarMatriz(Pq, estado);

        chavesCache.resize(m);
        valoresCache.resize(m);
        for(size_t i = 0; i < m; i++) {
            chavesCache[i] = aplicarMatriz(Pk, chaves[i]);
            valoresCache[i] = aplicarMatriz(Pv, chaves[i]);
        }
        pontosCache.resize(m);
        for(size_t i = 0; i < m; i++) {
            float dot = 0.0f;
            for(size_t j = 0; j < dimAtencao; j++) dot += consultaCache[j] * chavesCache[i][j];
            pontosCache[i] = dot * escala;
        }
        pesosCache = softmax(pontosCache);

        vector<float> saida(dimSaida, 0.0f);
        for(size_t i = 0; i < m; i++) {
            for(size_t j = 0; j < dimSaida; j++) {
                saida[j] += pesosCache[i] * valoresCache[i][j];
            }
        }
        return saida;
    }

    // propLote: entrada [T x dim] -> saida [T x dim]
    // mascara causal: token t atende a 0..t
    vector<vector<float>> propLote(const vector<vector<float>>& entrada) override {
        size_t T = entrada.size();
        if(T == 0) throw invalid_argument("[" + nome + "]: lote vazio");

        cacheLote.entrada = entrada;
        cacheLote.Q.resize(T);
        cacheLote.K.resize(T);
        cacheLote.V.resize(T);
        cacheLote.pesos.assign(T, vector<float>(T, 0.0f));

        for(size_t t = 0; t < T; t++) {
            cacheLote.Q[t] = aplicarMatriz(Pq, entrada[t]);
            cacheLote.K[t] = aplicarMatriz(Pk, entrada[t]);
            cacheLote.V[t] = aplicarMatriz(Pv, entrada[t]);
        }
        vector<vector<float>> saida(T, vector<float>(dimSaida, 0.0f));

        for(size_t t = 0; t < T; t++) {
            // pontos q[t] * k[s] para s <= t
            vector<float> pontos(t + 1);
            for(size_t s = 0; s <= t; s++) {
                float dot = 0.0f;
                for(size_t j = 0; j < dimAtencao; j++)
                    dot += cacheLote.Q[t][j] * cacheLote.K[s][j];
                pontos[s] = dot * escala;
            }
            vector<float> pesos = softmax(pontos); // [t+1]
            for(size_t s = 0; s <= t; s++)
                cacheLote.pesos[t][s] = pesos[s];

            for(size_t s = 0; s <= t; s++) {
                for(size_t j = 0; j < dimSaida; j++)
                    saida[t][j] += pesos[s] * cacheLote.V[s][j];
            }
        }
        return saida;
    }

    GradGenerico retroprop(const vector<float>& gradSaida) override {
        if(gradSaida.size() != dimSaida) throw invalid_argument("[" + nome + "]: dimensão do gradiente de saída incorreta");

        size_t m = pesosCache.size();

        vector<vector<float>> gradV(m, vector<float>(dimSaida));
        for(size_t i = 0; i < m; i++) {
            for(size_t j = 0; j < dimSaida; j++) {
                gradV[i][j] = pesosCache[i] * gradSaida[j];
            }
        }
        vector<float> gradPesos(m);
        for(size_t i = 0; i < m; i++) {
            float dot = 0.0f;
            for(size_t j = 0; j < dimSaida; j++) dot += gradSaida[j] * valoresCache[i][j];
            gradPesos[i] = dot;
        }
        vector<float> gradpontos(m);
        float soma = 0.0f;
        for(size_t i = 0; i < m; i++) soma += gradPesos[i] * pesosCache[i];
        for(size_t i = 0; i < m; i++) gradpontos[i] = pesosCache[i] * (gradPesos[i] - soma);
        for(size_t i = 0; i < m; i++) gradpontos[i] *= escala;

        vector<float> gradQ(dimAtencao, 0.0f);
        for(size_t i = 0; i < m; i++) {
            for(size_t j = 0; j < dimAtencao; j++) {
                gradQ[j] += gradpontos[i] * chavesCache[i][j];
            }
        }
        vector<vector<float>> gradK(m, vector<float>(dimAtencao));
        for(size_t i = 0; i < m; i++) {
            for(size_t j = 0; j < dimAtencao; j++) {
                gradK[i][j] = gradpontos[i] * consultaCache[j];
            }
        }
        for(size_t i = 0; i < dimAtencao; i++) {
            for(size_t j = 0; j < dim; j++) {
                gradPq[i][j] += gradQ[i] * entradaCache[j];
            }
        }
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
        vector<float> gradEstado(dim, 0.0f);
        for(size_t j = 0; j < dim; j++) {
            for(size_t i = 0; i < dimAtencao; i++) {
                gradEstado[j] += Pq[i][j] * gradQ[i];
            }
        }
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

    // retropropLote: gradSaida [T x dimSaida] -> gradEntrada [T x dim]
    // acumula gradPq, gradPk, gradPv
    vector<vector<float>> retropropLote(const vector<vector<float>>& gradSaida) override {
        size_t T = gradSaida.size();
        if(T == 0) throw invalid_argument("[" + nome + "]: lote vazio no retroprop");

        // gradV[t][s][j] = pesos[t][s] * gradSaida[t][j]
        // gradPesos[t][s] = gradSaida[t] · V[s]
        // gradpontos[t][s] via jacobiana softmax
        // gradQ[t] = soma_s(gradpontos[t][s] * K[s])
        // gradK[s] = soma_t(gradpontos[t][s] * Q[t])
        // gradV[s] = soma_t(pesos[t][s] * gradSaida[t])

        // gradK e gradV acumulados por posição de chave
        vector<vector<float>> gradK(T, vector<float>(dimAtencao, 0.0f));
        vector<vector<float>> gradVk(T, vector<float>(dimSaida, 0.0f));

        vector<vector<float>> gradEntrada(T, vector<float>(dim, 0.0f));

        for(size_t t = 0; t < T; t++) {
            size_t m = t + 1; // token t atende a 0..t

            // gradPesos[s] = gradSaida[t] * V[s]
            vector<float> gPesos(m);
            for(size_t s = 0; s < m; s++) {
                float dot = 0.0f;
                for(size_t j = 0; j < dimSaida; j++) {
                    dot += gradSaida[t][j] * cacheLote.V[s][j];
                }
                gPesos[s] = dot;
            }
            // acumula gradV[s] += pesos[t][s] * gradSaida[t]
            for(size_t s = 0; s < m; s++) {
                for(size_t j = 0; j < dimSaida; j++) {
                    gradVk[s][j] += cacheLote.pesos[t][s] * gradSaida[t][j];
                }
            }
            // jacobiana softmax
            vector<float> gPontos(m);
            float soma = 0.0f;
            for(size_t s = 0; s < m; s++) soma += gPesos[s] * cacheLote.pesos[t][s];
            for(size_t s = 0; s < m; s++) {
                gPontos[s] = cacheLote.pesos[t][s] * (gPesos[s] - soma) * escala;
            }
            // gradQ[t] = soma_s(gPontos[s] * K[s])
            vector<float> gQ(dimAtencao, 0.0f);
            for(size_t s = 0; s < m; s++) {
                for(size_t j = 0; j < dimAtencao; j++) {
                    gQ[j] += gPontos[s] * cacheLote.K[s][j];
                }
            }
            // acumula gradK[s] += gPontos[s] * Q[t]
            for(size_t s = 0; s < m; s++) {
                for(size_t j = 0; j < dimAtencao; j++) {
                    gradK[s][j] += gPontos[s] * cacheLote.Q[t][j];
                }
            }
            // acumula gradPq += gQ * entrada[t]
            for(size_t i = 0; i < dimAtencao; i++) {
                for(size_t j = 0; j < dim; j++) {
                    gradPq[i][j] += gQ[i] * cacheLote.entrada[t][j];
                }
            }
            // gradEntrada[t] via Pq^T * gQ
            for(size_t j = 0; j < dim; j++) {
                for(size_t i = 0; i < dimAtencao; i++) {
                    gradEntrada[t][j] += Pq[i][j] * gQ[i];
                }
            }
        }

        // acumula gradPk, gradPv e gradEntrada via K, V
        for(size_t s = 0; s < T; s++) {
            for(size_t i = 0; i < dimAtencao; i++) {
                for(size_t j = 0; j < dim; j++) {
                    gradPk[i][j] += gradK[s][i] * cacheLote.entrada[s][j];
                }
            }
            for(size_t i = 0; i < dimSaida; i++) {
                for(size_t j = 0; j < dim; j++) {
                    gradPv[i][j] += gradVk[s][i] * cacheLote.entrada[s][j];
                }
            }
            // gradEntrada[s] via Pk^T * gradK[s] + Pv^T * gradVk[s]
            for(size_t j = 0; j < dim; j++) {
                for(size_t i = 0; i < dimAtencao; i++) {
                    gradEntrada[s][j] += Pk[i][j] * gradK[s][i];
                }
                for(size_t i = 0; i < dimSaida; i++) {
                    gradEntrada[s][j] += Pv[i][j] * gradVk[s][i];
                }
            }
        }
        return gradEntrada;
    }

    void att(float taxaAprendizado) override {
        vector<float> biasZero(1, 0.0f);
        vector<float> gradBiasZero(1, 0.0f);

        if(otimQ) otimQ->att(Pq, gradPq, biasZero, gradBiasZero);
        else {
            for(size_t i = 0; i < dimAtencao; i++) {
                for(size_t j = 0; j < dim; j++) {
                    Pq[i][j] -= taxaAprendizado * gradPq[i][j];
                }
            }
        }
        if(otimK) otimK->att(Pk, gradPk, biasZero, gradBiasZero);
        else {
            for(size_t i = 0; i < dimAtencao; i++) {
                for(size_t j = 0; j < dim; j++) {
                    Pk[i][j] -= taxaAprendizado * gradPk[i][j];
                }
            }
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