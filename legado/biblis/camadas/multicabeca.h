// biblis/camadas/multicabeca.h
#pragma once
#include "camada.h"

// atenção multi-head — layout flat, matmul explícita
//
// fluxo (lote):
// entrada [T x dim]
//   -> H cabeças paralelas, cada uma com Wq/Wk/Wv próprios [dimCabeca x dim]
//   -> cada cabeça produz saída [T x dimCabeca] com máscara causal
//   -> concatena -> [T x dim]
//   -> Wo [dim x dim] -> saída [T x dim]
//
// dimCabeca = dim / numCabecas
//
// mudanças em relação à versão anterior:
//   - CacheLoteCabeca usa vetores flat (T*dC) em vez de vector<vector<float>>
//   - Q·K^T calculado como matmul contígua (SIMD-friendly)
//   - Wq/Wk/Wv/Wo armazenados como flat [linhas*colunas]
//   - gradientes também flat
//   - softmax inline sem alocação extra
//   - interface pública idêntica (propLote / retropropLote / att / etc.)

class MultiCabeca : public Camada {
public:
    size_t dim;
    size_t numCabecas;
    size_t dimCabeca;
    float  escala;

    // pesos flat: Wq[h] tem dimCabeca*dim floats, layout linha-maior
    // acesso: Wq[h][i*dim + j]  (i = linha saída, j = coluna entrada)
    vector<vector<float>> Wq;   // [numCabecas][dimCabeca * dim]
    vector<vector<float>> Wk;
    vector<vector<float>> Wv;
    vector<vector<float>> gradWq;
    vector<vector<float>> gradWk;
    vector<vector<float>> gradWv;

    // projeção de saída flat: [dim * dim]
    vector<float> Wo;       // [dim * dim]
    vector<float> gradWo;
    vector<float> biasO;    // [dim]
    vector<float> gradBiasO;

    // otimizadores por cabeça + saída
    vector<unique_ptr<Otimizador>> otimQ;
    vector<unique_ptr<Otimizador>> otimK;
    vector<unique_ptr<Otimizador>> otimV;
    unique_ptr<Otimizador> otimO;

    // cache lote — tudo flat, sem alocação por token
    struct CacheLoteCabeca {
        vector<float> Q;     // [T * dimCabeca]
        vector<float> K;
        vector<float> V;
        vector<float> pesos; // [T * T] — só triângulo inferior usado
        vector<float> saida; // [T * dimCabeca]
    };
    vector<CacheLoteCabeca> cacheCabecas; // [numCabecas]

    // entrada e concat em flat também
    vector<float> entradaCache; // [T * dim]
    vector<float> concatCache;  // [T * dim]
    size_t T_cache = 0;

    // ----------------------------------------------------------------
    // construtor
    // ----------------------------------------------------------------
    MultiCabeca(size_t dim, size_t numCabecas, const string& nome = "multicabeca")
        : Camada(nome), dim(dim), numCabecas(numCabecas),
          dimCabeca(dim / numCabecas),
          escala(1.0f / sqrtf((float)(dim / numCabecas))) {

        if(dim % numCabecas != 0)
            throw invalid_argument("[" + nome + "]: dim deve ser divisível por numCabecas");

        tipo = "MultiCabeca";

        Wq.resize(numCabecas);
        Wk.resize(numCabecas);
        Wv.resize(numCabecas);
        gradWq.resize(numCabecas);
        gradWk.resize(numCabecas);
        gradWv.resize(numCabecas);

        mt19937 rng(42);
        float limXavier = sqrtf(6.0f / (float)(dimCabeca + dim));
        uniform_real_distribution<float> dist(-limXavier, limXavier);

        for(size_t h = 0; h < numCabecas; h++) {
            size_t n = dimCabeca * dim;
            Wq[h].resize(n); for(auto& v : Wq[h]) v = dist(rng);
            Wk[h].resize(n); for(auto& v : Wk[h]) v = dist(rng);
            Wv[h].resize(n); for(auto& v : Wv[h]) v = dist(rng);
            gradWq[h].assign(n, 0.0f);
            gradWk[h].assign(n, 0.0f);
            gradWv[h].assign(n, 0.0f);
        }

        float limWo = sqrtf(6.0f / (float)(dim + dim));
        uniform_real_distribution<float> distWo(-limWo, limWo);
        size_t nWo = dim * dim;
        Wo.resize(nWo);     for(auto& v : Wo) v = distWo(rng);
        gradWo.assign(nWo, 0.0f);
        biasO.assign(dim, 0.0f);
        gradBiasO.assign(dim, 0.0f);

        cacheCabecas.resize(numCabecas);
    }

    // prop token-a-token não implementado — modelo usa propLote
    vector<float> prop(const vector<float>& entrada) override {
        throw runtime_error("[" + nome + "]: use propLote");
    }

    GradGenerico retroprop(const vector<float>& grad) override {
        throw runtime_error("[" + nome + "]: use retropropLote");
    }

    // ----------------------------------------------------------------
    // propLote: entrada [T x dim] (vector<vector>) -> saída [T x dim]
    // interface idêntica ao original
    // ----------------------------------------------------------------
    vector<vector<float>> propLote(const vector<vector<float>>& entrada) override {
        size_t T = entrada.size();
        if(T == 0) throw invalid_argument("[" + nome + "]: lote vazio");

        T_cache = T;

        // copia entrada para flat [T * dim]
        entradaCache.resize(T * dim);
        for(size_t t = 0; t < T; t++)
            for(size_t j = 0; j < dim; j++)
                entradaCache[t * dim + j] = entrada[t][j];

        // concat flat [T * dim] — preenchido por offset de cabeça
        concatCache.assign(T * dim, 0.0f);

        for(size_t h = 0; h < numCabecas; h++) {
            auto& cc = cacheCabecas[h];
            cc.Q.resize(T * dimCabeca);
            cc.K.resize(T * dimCabeca);
            cc.V.resize(T * dimCabeca);
            cc.pesos.assign(T * T, 0.0f);
            cc.saida.assign(T * dimCabeca, 0.0f);

            const float* Wqh = Wq[h].data();
            const float* Wkh = Wk[h].data();
            const float* Wvh = Wv[h].data();

            // projeta Q, K, V: [T x dim] × [dimCabeca x dim]^T -> [T x dimCabeca]
            // Wqh layout: [i*dim + j], i=0..dimCabeca-1, j=0..dim-1
            for(size_t t = 0; t < T; t++) {
                const float* et = &entradaCache[t * dim];
                float* qt = &cc.Q[t * dimCabeca];
                float* kt = &cc.K[t * dimCabeca];
                float* vt = &cc.V[t * dimCabeca];

                for(size_t i = 0; i < dimCabeca; i++) {
                    const float* wqi = &Wqh[i * dim];
                    const float* wki = &Wkh[i * dim];
                    const float* wvi = &Wvh[i * dim];
                    float sq = 0.0f, sk = 0.0f, sv = 0.0f;
                    for(size_t j = 0; j < dim; j++) {
                        sq += wqi[j] * et[j];
                        sk += wki[j] * et[j];
                        sv += wvi[j] * et[j];
                    }
                    qt[i] = sq;
                    kt[i] = sk;
                    vt[i] = sv;
                }
            }

            // atenção causal: pontos[t][s] = Q[t] · K[s] * escala  (s <= t)
            // depois softmax em linha, depois soma ponderada V
            for(size_t t = 0; t < T; t++) {
                const float* qt = &cc.Q[t * dimCabeca];
                float* pesos_t  = &cc.pesos[t * T];

                // produto escalar Q[t] · K[s] para s=0..t
                float maxVal = -1e30f;
                for(size_t s = 0; s <= t; s++) {
                    const float* ks = &cc.K[s * dimCabeca];
                    float dot = 0.0f;
                    for(size_t j = 0; j < dimCabeca; j++)
                        dot += qt[j] * ks[j];
                    dot *= escala;
                    pesos_t[s] = dot;
                    if(dot > maxVal) maxVal = dot;
                }

                // softmax estável inline — sem alocação
                float soma = 0.0f;
                for(size_t s = 0; s <= t; s++) {
                    pesos_t[s] = expf(pesos_t[s] - maxVal);
                    soma += pesos_t[s];
                }
                float invSoma = 1.0f / soma;
                for(size_t s = 0; s <= t; s++)
                    pesos_t[s] *= invSoma;

                // soma ponderada: saida[t] += pesos[t][s] * V[s]
                float* st = &cc.saida[t * dimCabeca];
                for(size_t s = 0; s <= t; s++) {
                    float p = pesos_t[s];
                    const float* vs = &cc.V[s * dimCabeca];
                    for(size_t j = 0; j < dimCabeca; j++)
                        st[j] += p * vs[j];
                }
            }

            // copia saída da cabeça h no offset correspondente de concatCache
            size_t offset = h * dimCabeca;
            for(size_t t = 0; t < T; t++) {
                const float* st = &cc.saida[t * dimCabeca];
                float* ct = &concatCache[t * dim + offset];
                for(size_t j = 0; j < dimCabeca; j++)
                    ct[j] = st[j];
            }
        }

        // projeção de saída Wo: [T x dim] × [dim x dim] -> [T x dim]
        // Wo layout: [i*dim + j]
        vector<vector<float>> saida(T, vector<float>(dim));
        const float* Wod = Wo.data();
        for(size_t t = 0; t < T; t++) {
            const float* ct = &concatCache[t * dim];
            float* st = saida[t].data();
            for(size_t i = 0; i < dim; i++) {
                const float* woi = &Wod[i * dim];
                float acc = biasO[i];
                for(size_t j = 0; j < dim; j++)
                    acc += woi[j] * ct[j];
                st[i] = acc;
            }
        }
        return saida;
    }

    // ----------------------------------------------------------------
    // retropropLote: gradSaida [T x dim] -> gradEntrada [T x dim]
    // ----------------------------------------------------------------
    vector<vector<float>> retropropLote(const vector<vector<float>>& gradSaida) override {
        size_t T = gradSaida.size();

        // gradSaida para flat temporário
        vector<float> gradSaidaFlat(T * dim);
        for(size_t t = 0; t < T; t++)
            for(size_t i = 0; i < dim; i++)
                gradSaidaFlat[t * dim + i] = gradSaida[t][i];

        // --- retroprop de Wo ---
        // gradConcat[t][j] = sum_i(Wo[i][j] * gradSaida[t][i])  (Wo^T * grad)
        // gradWo[i][j]     += gradSaida[t][i] * concat[t][j]
        vector<float> gradConcatFlat(T * dim, 0.0f);
        const float* Wod = Wo.data();
        float* gWod = gradWo.data();

        for(size_t t = 0; t < T; t++) {
            const float* gs  = &gradSaidaFlat[t * dim];
            const float* ct  = &concatCache[t * dim];
            float*       gct = &gradConcatFlat[t * dim];

            for(size_t i = 0; i < dim; i++) {
                float gsi = gs[i];
                gradBiasO[i] += gsi;
                const float* woi = &Wod[i * dim];
                float* gwoi = &gWod[i * dim];
                for(size_t j = 0; j < dim; j++) {
                    gwoi[j] += gsi * ct[j];   // gradWo
                    gct[j]  += woi[j] * gsi;  // gradConcat (Wo^T * gs)
                }
            }
        }

        // gradiente total sobre a entrada (acumulado de todas as cabeças)
        vector<float> gradEntradaFlat(T * dim, 0.0f);

        for(size_t h = 0; h < numCabecas; h++) {
            auto& cc = cacheCabecas[h];
            size_t offset = h * dimCabeca;

            const float* Wqh = Wq[h].data();
            const float* Wkh = Wk[h].data();
            const float* Wvh = Wv[h].data();
            float* gWqh = gradWq[h].data();
            float* gWkh = gradWk[h].data();
            float* gWvh = gradWv[h].data();

            // fatia gradConcat para esta cabeça: gradCabeca[t] = gradConcatFlat[t][offset..+dimCabeca)
            // gradK[s], gradVk[s]: flat [T * dimCabeca]
            vector<float> gradK(T * dimCabeca, 0.0f);
            vector<float> gradVk(T * dimCabeca, 0.0f);

            for(size_t t = 0; t < T; t++) {
                size_t m = t + 1;
                const float* gc_t  = &gradConcatFlat[t * dim + offset]; // gradCabeca[t]
                const float* pesos_t = &cc.pesos[t * T];

                // gPesos[s] = gc_t · V[s]
                // gV[s]     += pesos[t][s] * gc_t
                vector<float> gPesos(m, 0.0f);
                for(size_t s = 0; s < m; s++) {
                    const float* vs = &cc.V[s * dimCabeca];
                    float* gvs = &gradVk[s * dimCabeca];
                    float p = pesos_t[s];
                    float dot = 0.0f;
                    for(size_t j = 0; j < dimCabeca; j++) {
                        dot      += gc_t[j] * vs[j];
                        gvs[j]   += p * gc_t[j];
                    }
                    gPesos[s] = dot;
                }

                // jacobiana softmax: gPontos[s] = pesos[s] * (gPesos[s] - soma) * escala
                float soma = 0.0f;
                for(size_t s = 0; s < m; s++) soma += gPesos[s] * pesos_t[s];
                vector<float> gPontos(m);
                for(size_t s = 0; s < m; s++)
                    gPontos[s] = pesos_t[s] * (gPesos[s] - soma) * escala;

                // gQ[t] = sum_s(gPontos[s] * K[s])
                // gK[s] += gPontos[s] * Q[t]
                const float* qt = &cc.Q[t * dimCabeca];
                vector<float> gQ(dimCabeca, 0.0f);
                for(size_t s = 0; s < m; s++) {
                    const float* ks = &cc.K[s * dimCabeca];
                    float* gks = &gradK[s * dimCabeca];
                    float gp = gPontos[s];
                    for(size_t j = 0; j < dimCabeca; j++) {
                        gQ[j]  += gp * ks[j];
                        gks[j] += gp * qt[j];
                    }
                }

                // gradWq[h] += gQ * entrada[t]^T
                // gradEntrada[t] += Wq^T * gQ
                const float* et = &entradaCache[t * dim];
                float* get = &gradEntradaFlat[t * dim];
                for(size_t i = 0; i < dimCabeca; i++) {
                    float gqi = gQ[i];
                    float* gwqi = &gWqh[i * dim];
                    const float* wqi = &Wqh[i * dim];
                    for(size_t j = 0; j < dim; j++) {
                        gwqi[j] += gqi * et[j];
                        get[j]  += wqi[j] * gqi;
                    }
                }
            }

            // gradWk, gradWv e gradEntrada via K, V — varrendo por posição-chave s
            for(size_t s = 0; s < T; s++) {
                const float* es  = &entradaCache[s * dim];
                float*       ges = &gradEntradaFlat[s * dim];
                const float* gks = &gradK[s * dimCabeca];
                const float* gvs = &gradVk[s * dimCabeca];

                for(size_t i = 0; i < dimCabeca; i++) {
                    float gki = gks[i];
                    float gvi = gvs[i];
                    float* gwki = &gWkh[i * dim];
                    float* gwvi = &gWvh[i * dim];
                    const float* wki = &Wkh[i * dim];
                    const float* wvi = &Wvh[i * dim];
                    for(size_t j = 0; j < dim; j++) {
                        gwki[j] += gki * es[j];
                        gwvi[j] += gvi * es[j];
                        ges[j]  += wki[j] * gki + wvi[j] * gvi;
                    }
                }
            }
        }

        // converte gradEntradaFlat de volta para vector<vector<float>>
        vector<vector<float>> gradEntrada(T, vector<float>(dim));
        for(size_t t = 0; t < T; t++)
            for(size_t j = 0; j < dim; j++)
                gradEntrada[t][j] = gradEntradaFlat[t * dim + j];

        return gradEntrada;
    }

    // ----------------------------------------------------------------
    // att: atualiza pesos via otimizadores
    // ----------------------------------------------------------------
    void att(float taxaAprendizado) override {
        // helper: converte flat -> vector<vector> (shape), chama otimizador, converte de volta
        // Os otimizadores existentes esperam vector<vector<float>> — empacota e desempacota
        auto flatParaMat = [&](const vector<float>& flat, size_t linhas, size_t colunas)
            -> vector<vector<float>> {
            vector<vector<float>> M(linhas, vector<float>(colunas));
            for(size_t i = 0; i < linhas; i++)
                for(size_t j = 0; j < colunas; j++)
                    M[i][j] = flat[i * colunas + j];
            return M;
        };
        auto matParaFlat = [&](const vector<vector<float>>& M, size_t linhas, size_t colunas,
                               vector<float>& flat) {
            for(size_t i = 0; i < linhas; i++)
                for(size_t j = 0; j < colunas; j++)
                    flat[i * colunas + j] = M[i][j];
        };

        vector<float> biasZero(1, 0.0f);
        vector<float> gradBiasZero(1, 0.0f);

        for(size_t h = 0; h < numCabecas; h++) {
            auto atualizarCabeca = [&](vector<float>& W, vector<float>& gW,
                                       unique_ptr<Otimizador>& otim) {
                if(otim) {
                    auto Wmat  = flatParaMat(W,  dimCabeca, dim);
                    auto gWmat = flatParaMat(gW, dimCabeca, dim);
                    otim->att(Wmat, gWmat, biasZero, gradBiasZero);
                    matParaFlat(Wmat, dimCabeca, dim, W);
                } else {
                    size_t n = dimCabeca * dim;
                    for(size_t k = 0; k < n; k++)
                        W[k] -= taxaAprendizado * gW[k];
                }
            };

            if(h < otimQ.size()) atualizarCabeca(Wq[h], gradWq[h], otimQ[h]);
            else { size_t n = dimCabeca * dim; for(size_t k=0;k<n;k++) Wq[h][k] -= taxaAprendizado * gradWq[h][k]; }

            if(h < otimK.size()) atualizarCabeca(Wk[h], gradWk[h], otimK[h]);
            else { size_t n = dimCabeca * dim; for(size_t k=0;k<n;k++) Wk[h][k] -= taxaAprendizado * gradWk[h][k]; }

            if(h < otimV.size()) atualizarCabeca(Wv[h], gradWv[h], otimV[h]);
            else { size_t n = dimCabeca * dim; for(size_t k=0;k<n;k++) Wv[h][k] -= taxaAprendizado * gradWv[h][k]; }
        }

        if(otimO) {
            auto Wmat  = flatParaMat(Wo,     dim, dim);
            auto gWmat = flatParaMat(gradWo, dim, dim);
            otimO->att(Wmat, gWmat, biasO, gradBiasO);
            matParaFlat(Wmat, dim, dim, Wo);
            // biasO já atualizado in-place pelo otimizador
        } else {
            size_t n = dim * dim;
            for(size_t k = 0; k < n; k++)
                Wo[k] -= taxaAprendizado * gradWo[k];
            for(size_t i = 0; i < dim; i++)
                biasO[i] -= taxaAprendizado * gradBiasO[i];
        }
    }

    void zerarGradientes() override {
        for(size_t h = 0; h < numCabecas; h++) {
            fill(gradWq[h].begin(), gradWq[h].end(), 0.0f);
            fill(gradWk[h].begin(), gradWk[h].end(), 0.0f);
            fill(gradWv[h].begin(), gradWv[h].end(), 0.0f);
        }
        fill(gradWo.begin(),     gradWo.end(),     0.0f);
        fill(gradBiasO.begin(),  gradBiasO.end(),  0.0f);
    }

    bool temParametros() const override { return true; }
    size_t numParametros() const override {
        return numCabecas * 3 * dimCabeca * dim + dim * dim + dim;
    }

    void defOtimizadores(
        vector<unique_ptr<Otimizador>> oQ,
        vector<unique_ptr<Otimizador>> oK,
        vector<unique_ptr<Otimizador>> oV,
        unique_ptr<Otimizador> oO
    ) {
        otimQ = std::move(oQ);
        otimK = std::move(oK);
        otimV = std::move(oV);
        otimO = std::move(oO);
    }

    // ----------------------------------------------------------------
    // salvar / carregar — converte flat <-> linhas para manter
    // compatibilidade com o formato binário já gravado em disco
    // ----------------------------------------------------------------
    void salvar(const string& prefixo) const override {
        // salva Wo (dim x dim)
        {
            ofstream a(prefixo + "_Wo.bin", ios::binary);
            if(!a) throw runtime_error("[" + nome + "]: falha ao salvar Wo");
            size_t d = dim;
            a.write(reinterpret_cast<const char*>(&d), sizeof(d));
            // escreve linha por linha — compatível com formato antigo
            for(size_t i = 0; i < dim; i++)
                a.write(reinterpret_cast<const char*>(&Wo[i * dim]), dim * sizeof(float));
            a.write(reinterpret_cast<const char*>(biasO.data()), dim * sizeof(float));
        }
        for(size_t h = 0; h < numCabecas; h++) {
            auto salvarFlat = [&](const vector<float>& flat, size_t linhas, size_t colunas,
                                  const string& sufixo) {
                ofstream a(prefixo + sufixo + to_string(h) + ".bin", ios::binary);
                if(!a) throw runtime_error("[" + nome + "]: falha ao salvar " + sufixo);
                a.write(reinterpret_cast<const char*>(&linhas), sizeof(linhas));
                a.write(reinterpret_cast<const char*>(&colunas), sizeof(colunas));
                // flat já está em linha-maior — uma escrita só
                a.write(reinterpret_cast<const char*>(flat.data()), linhas * colunas * sizeof(float));
            };
            salvarFlat(Wq[h], dimCabeca, dim, "_Wq");
            salvarFlat(Wk[h], dimCabeca, dim, "_Wk");
            salvarFlat(Wv[h], dimCabeca, dim, "_Wv");
        }
    }

    void carregar(const string& prefixo) override {
        {
            ifstream a(prefixo + "_Wo.bin", ios::binary);
            if(!a) throw runtime_error("[" + nome + "]: falha ao carregar Wo");
            size_t d;
            a.read(reinterpret_cast<char*>(&d), sizeof(d));
            Wo.resize(d * d);
            for(size_t i = 0; i < d; i++)
                a.read(reinterpret_cast<char*>(&Wo[i * d]), d * sizeof(float));
            biasO.resize(d);
            a.read(reinterpret_cast<char*>(biasO.data()), d * sizeof(float));
        }
        for(size_t h = 0; h < numCabecas; h++) {
            auto carregarFlat = [&](vector<float>& flat, const string& sufixo) {
                ifstream a(prefixo + sufixo + to_string(h) + ".bin", ios::binary);
                if(!a) throw runtime_error("[" + nome + "]: falha ao carregar " + sufixo);
                size_t linhas, colunas;
                a.read(reinterpret_cast<char*>(&linhas), sizeof(linhas));
                a.read(reinterpret_cast<char*>(&colunas), sizeof(colunas));
                flat.resize(linhas * colunas);
                a.read(reinterpret_cast<char*>(flat.data()), linhas * colunas * sizeof(float));
            };
            carregarFlat(Wq[h], "_Wq");
            carregarFlat(Wk[h], "_Wk");
            carregarFlat(Wv[h], "_Wv");
        }
        gradWo.assign(dim * dim, 0.0f);
        gradBiasO.assign(dim, 0.0f);
        for(size_t h = 0; h < numCabecas; h++) {
            gradWq[h].assign(dimCabeca * dim, 0.0f);
            gradWk[h].assign(dimCabeca * dim, 0.0f);
            gradWv[h].assign(dimCabeca * dim, 0.0f);
        }
    }

};
