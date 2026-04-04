// biblis/camadas/bloco.h
#pragma once
#include "camada.h"
#include "atencao.h"
#include "densa.h"
#include "norm.h"

// bloco transformer padrão com conexões residuais

// fluxo token-a-token:
// x -> norm1 -> atencao(x, chaves) -> + x  -> x1
// x1 -> norm2 -> oculta -> + x1 -> saida

// fluxo lote [T x dim]:
// X -> norm1 -> atencao_lote(X) -> + X -> X1
// X1 -> norm2 -> oculta -> + X1 -> saida

class BlocoTransformer : public Camada {
public:
    size_t dim;
    size_t dimAtencao;
    size_t dimoculta;

    CamadaNorm norm1;
    CamadaAtencao atencao;
    CamadaNorm norm2;
    Densa oculta1;
    Densa oculta2;

    // cache token-a-token
    vector<float> entradaCache;
    vector<float> x1Cache;
    vector<float> normSaida1;
    vector<float> normSaida2;
    vector<float> atencaoSaida;
    vector<float> oculta1Saida;
    vector<vector<float>> chavesCache;

    // cache lote: estado por token
    struct CacheLoteBloco {
        vector<vector<float>> entrada; // [T x dim]
        vector<vector<float>> normSaida1; // [T x dim]
        vector<vector<float>> atencaoSaida; // [T x dim]
        vector<vector<float>> x1; // [T x dim]
        vector<vector<float>> normSaida2; // [T x dim]
        vector<vector<float>> oculta1Saida; // [T x dimoculta]
    } cacheLoteBloco;

    BlocoTransformer(
        size_t dim,
        size_t dimAtencao,
        size_t dimoculta = 0,
        const string& ativacaooculta = "relu",
        const string& nome = "bloco"
    )
    : Camada(nome),
          dim(dim),
          dimAtencao(dimAtencao),
          dimoculta(dimoculta > 0 ? dimoculta : 4 * dim),
          norm1(dim, 1e-5f, nome + "_norm1"),
          atencao(dim, dimAtencao, dim, nome + "_atencao"),
          norm2(dim, 1e-5f, nome + "_norm2"),
          oculta1(dim, (dimoculta > 0 ? dimoculta : 4 * dim), ativacaooculta, true, nome + "_oculta1"),
          oculta2((dimoculta > 0 ? dimoculta : 4 * dim), dim, "linear", true, nome + "_oculta2") {
        tipo = "BlocoTransformer";
    }

    vector<float> prop(const vector<float>& entrada) override {
        return prop(entrada, {entrada});
    }

    vector<float> prop(
        const vector<float>& entrada,
        const vector<vector<float>>& chaves
    ) {
        if(entrada.size() != dim)
            throw invalid_argument("[" + nome + "]: dimensão de entrada incorreta");

        entradaCache = entrada;
        chavesCache  = chaves;

        normSaida1 = norm1.prop(entrada);
        atencaoSaida = atencao.prop(normSaida1, chaves);

        x1Cache.resize(dim);
        for(size_t i = 0; i < dim; i++)
            x1Cache[i] = entrada[i] + atencaoSaida[i];

        normSaida2 = norm2.prop(x1Cache);
        oculta1Saida  = oculta1.prop(normSaida2);
        vector<float> oculta2Saida = oculta2.prop(oculta1Saida);

        vector<float> saida(dim);
        for(size_t i = 0; i < dim; i++)
            saida[i] = x1Cache[i] + oculta2Saida[i];
        return saida;
    }
    // propLote: entrada [T x dim] -> saida [T x dim]
    // atenção causal sobre toda a sequência de uma vez
    vector<vector<float>> propLote(const vector<vector<float>>& entrada) override {
        size_t T = entrada.size();
        if(T == 0) throw invalid_argument("[" + nome + "]: lote vazio");

        cacheLoteBloco.entrada = entrada;

        // norm1 sobre lote inteiro(cache por token dentro de norm1)
        cacheLoteBloco.normSaida1 = norm1.propLote(entrada);

        // atenção causal sobre lote inteiro
        cacheLoteBloco.atencaoSaida = atencao.propLote(cacheLoteBloco.normSaida1);

        // residual: x1 = entrada + atencaoSaida
        cacheLoteBloco.x1.resize(T, vector<float>(dim));
        for(size_t t = 0; t < T; t++) {
            for(size_t i = 0; i < dim; i++) {
                cacheLoteBloco.x1[t][i] = entrada[t][i] + cacheLoteBloco.atencaoSaida[t][i];
            }
        }
        // norm2 sobre lote inteiro (cache por token dentro de norm2)
        cacheLoteBloco.normSaida2 = norm2.propLote(cacheLoteBloco.x1);

        // oculta por token(Densa ja acumula gradientes corretamente token a token)
        cacheLoteBloco.oculta1Saida.resize(T);
        vector<vector<float>> saida(T, vector<float>(dim));
        for(size_t t = 0; t < T; t++) {
            cacheLoteBloco.oculta1Saida[t] = oculta1.prop(cacheLoteBloco.normSaida2[t]);
            vector<float> o2 = oculta2.prop(cacheLoteBloco.oculta1Saida[t]);
            for(size_t i = 0; i < dim; i++) {
                saida[t][i] = cacheLoteBloco.x1[t][i] + o2[i];
            }
        }
        return saida;
    }

    GradGenerico retroprop(const vector<float>& gradSaida) override {
        if(gradSaida.size() != dim)
            throw invalid_argument("[" + nome + "]: dimensão do gradiente incorreta");

        vector<float> gradX1(gradSaida);

        auto goculta2 = oculta2.retroprop(gradSaida);
        auto goculta1 = oculta1.retroprop(goculta2.vetor);
        auto gNorm2 = norm2.retroprop(goculta1.vetor);

        for(size_t i = 0; i < dim; i++)
            gradX1[i] += gNorm2.vetor[i];

        vector<float> gradEntrada(gradX1);

        auto gAtencao = atencao.retroprop(gradX1);
        auto gNorm1 = norm1.retroprop(gAtencao.gradEstado);

        for(size_t i = 0; i < dim; i++) {
            gradEntrada[i] += gNorm1.vetor[i];
        }
        return GradGenerico(gradEntrada, gAtencao.gradChaves);
    }

    // retropropLote: gradSaida [T x dim] -> gradEntrada [T x dim]
    vector<vector<float>> retropropLote(const vector<vector<float>>& gradSaida) override {
        size_t T = gradSaida.size();
        if(T == 0) throw invalid_argument("[" + nome + "]: lote vazio no retroprop");

        // residual oculta: gradX1[t] = gradSaida[t] + grad_via_oculta[t]
        // oculta foi processada token a token, retroprop token a token tambem
        vector<vector<float>> gradNorm2(T, vector<float>(dim));
        for(size_t t = 0; t < T; t++) {
            auto go2 = oculta2.retroprop(gradSaida[t]);
            auto go1 = oculta1.retroprop(go2.vetor);
            gradNorm2[t] = go1.vetor;
        }
        // norm2 retroprop sobre lote(usa cacheLoteNorm de norm2)
        vector<vector<float>> gradX1 = norm2.retropropLote(gradNorm2);
        for(size_t t = 0; t < T; t++) {
            for(size_t i = 0; i < dim; i++) {
                gradX1[t][i] += gradSaida[t][i]; // residual
            }
        }
        // atenção causal retroprop sobre lote
        vector<vector<float>> gradNorm1 = atencao.retropropLote(gradX1);

        // norm1 retroprop sobre lote(usa cacheLoteNorm de norm1)
        vector<vector<float>> gradEntradaNorm1 = norm1.retropropLote(gradNorm1);

        // residual atenção
        vector<vector<float>> gradEntrada(T, vector<float>(dim));
        for(size_t t = 0; t < T; t++) {
            for(size_t i = 0; i < dim; i++) {
                gradEntrada[t][i] = gradX1[t][i] + gradEntradaNorm1[t][i];
            }
        }
        return gradEntrada;
    }

    void att(float taxaAprendizado) override {
        norm1.att(taxaAprendizado);
        atencao.att(taxaAprendizado);
        norm2.att(taxaAprendizado);
        oculta1.att(taxaAprendizado);
        oculta2.att(taxaAprendizado);
    }

    void zerarGradientes() override {
        norm1.zerarGradientes();
        atencao.zerarGradientes();
        norm2.zerarGradientes();
        oculta1.zerarGradientes();
        oculta2.zerarGradientes();
    }

    bool temParametros() const override { return true; }

    size_t numParametros() const override {
        return norm1.numParametros()
        + atencao.numParametros()
        + norm2.numParametros()
        + oculta1.numParametros()
        + oculta2.numParametros();
    }

    void defOtimizadores(
        unique_ptr<Otimizador> oNorm1,
        unique_ptr<Otimizador> oAtencao,
        unique_ptr<Otimizador> oNorm2,
        unique_ptr<Otimizador> ooculta1,
        unique_ptr<Otimizador> ooculta2
    ) {
        norm1.defOtimizador(std::move(oNorm1));
        norm2.defOtimizador(std::move(oNorm2));
        oculta1.defOtimizador(std::move(ooculta1));
        oculta2.defOtimizador(std::move(ooculta2));
        atencao.defOtimizador(std::move(oAtencao));
    }

    void salvar(const string& prefixo) const override {
        norm1.salvar(prefixo + "_norm1.bin");
        atencao.salvar(prefixo + "_atencao.bin");
        norm2.salvar(prefixo + "_norm2.bin");
        oculta1.salvar(prefixo + "_oculta1.bin");
        oculta2.salvar(prefixo + "_oculta2.bin");
    }

    void carregar(const string& prefixo) override {
        norm1.carregar(prefixo + "_norm1.bin");
        atencao.carregar(prefixo + "_atencao.bin");
        norm2.carregar(prefixo + "_norm2.bin");
        oculta1.carregar(prefixo + "_oculta1.bin");
        oculta2.carregar(prefixo + "_oculta2.bin");
    }
};