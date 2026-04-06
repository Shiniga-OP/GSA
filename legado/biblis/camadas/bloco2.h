// biblis/camadas/bloco2.h
#pragma once
#include "camada.h"
#include "multicabeca.h"
#include "densa.h"
#include "norm.h"

// bloco transformer com atenção multi-head
//
// fluxo lote [T x dim]:
// X -> norm1 -> MultiCabeca -> + X -> X1
// X1 -> norm2 -> FFN (oculta1 -> oculta2) -> + X1 -> saída
//
// pre-norm: normaliza antes de cada sub-camada (mais estável que post-norm)

class BlocoTransformerV2 : public Camada {
public:
    size_t dim;
    size_t numCabecas;
    size_t dimOculta;

    CamadaNorm norm1;
    MultiCabeca atencao;
    CamadaNorm norm2;
    Densa oculta1;
    Densa oculta2;

    // cache lote
    struct CacheBloco {
        vector<vector<float>> entrada;      // [T x dim]
        vector<vector<float>> normSaida1;   // [T x dim]
        vector<vector<float>> atencaoSaida; // [T x dim]
        vector<vector<float>> x1;           // [T x dim]
        vector<vector<float>> normSaida2;   // [T x dim]
        vector<vector<float>> ocultaSaida1; // [T x dimOculta]
        vector<vector<float>> ocultaSaida2; // [T x dim]
    } cache;

    BlocoTransformerV2(
        size_t dim,
        size_t numCabecas,
        size_t dimOculta = 0,
        const string& ativacaoOculta = "gelu",
        const string& nome = "bloco2"
    )
    : Camada(nome),
      dim(dim),
      numCabecas(numCabecas),
      dimOculta(dimOculta > 0 ? dimOculta : 4 * dim),
      norm1(dim, 1e-5f, nome + "_norm1"),
      atencao(dim, numCabecas, nome + "_atencao"),
      norm2(dim, 1e-5f, nome + "_norm2"),
      oculta1(dim, (dimOculta > 0 ? dimOculta : 4 * dim), ativacaoOculta, true, nome + "_oculta1"),
      oculta2((dimOculta > 0 ? dimOculta : 4 * dim), dim, "linear", true, nome + "_oculta2") {
        tipo = "BlocoTransformerV2";
    }

    vector<float> prop(const vector<float>& entrada) override {
        throw runtime_error("[" + nome + "]: use propLote");
    }

    GradGenerico retroprop(const vector<float>& grad) override {
        throw runtime_error("[" + nome + "]: use retropropLote");
    }

    vector<vector<float>> propLote(const vector<vector<float>>& entrada) override {
        size_t T = entrada.size();
        if(T == 0) throw invalid_argument("[" + nome + "]: lote vazio");

        cache.entrada = entrada;

        // pre-norm + atenção multi-head
        cache.normSaida1 = norm1.propLote(entrada);
        cache.atencaoSaida = atencao.propLote(cache.normSaida1);

        // residual
        cache.x1.resize(T, vector<float>(dim));
        for(size_t t = 0; t < T; t++)
            for(size_t i = 0; i < dim; i++)
                cache.x1[t][i] = entrada[t][i] + cache.atencaoSaida[t][i];

        // pre-norm + FFN
        cache.normSaida2 = norm2.propLote(cache.x1);

        cache.ocultaSaida1.resize(T);
        cache.ocultaSaida2.resize(T);
        vector<vector<float>> saida(T, vector<float>(dim));

        for(size_t t = 0; t < T; t++) {
            cache.ocultaSaida1[t] = oculta1.prop(cache.normSaida2[t]);
            cache.ocultaSaida2[t] = oculta2.prop(cache.ocultaSaida1[t]);

            // residual
            for(size_t i = 0; i < dim; i++)
                saida[t][i] = cache.x1[t][i] + cache.ocultaSaida2[t][i];
        }
        return saida;
    }

    // retropropLote: gradSaida [T x dim] -> gradEntrada [T x dim]
    vector<vector<float>> retropropLote(const vector<vector<float>>& gradSaida) override {
        size_t T = gradSaida.size();
        if(T == 0) throw invalid_argument("[" + nome + "]: lote vazio no retroprop");

        // retroprop FFN token a token
        vector<vector<float>> gradNorm2(T, vector<float>(dim));
        for(size_t t = 0; t < T; t++) {
            auto go2 = oculta2.retroprop(gradSaida[t]);
            auto go1 = oculta1.retroprop(go2.vetor);
            gradNorm2[t] = go1.vetor;
        }

        // retroprop norm2
        vector<vector<float>> gradX1 = norm2.retropropLote(gradNorm2);

        // residual FFN
        for(size_t t = 0; t < T; t++)
            for(size_t i = 0; i < dim; i++)
                gradX1[t][i] += gradSaida[t][i];

        // retroprop atenção multi-head
        vector<vector<float>> gradNorm1 = atencao.retropropLote(gradX1);

        // retroprop norm1
        vector<vector<float>> gradEntradaNorm1 = norm1.retropropLote(gradNorm1);

        // residual atenção
        vector<vector<float>> gradEntrada(T, vector<float>(dim));
        for(size_t t = 0; t < T; t++)
            for(size_t i = 0; i < dim; i++)
                gradEntrada[t][i] = gradX1[t][i] + gradEntradaNorm1[t][i];

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

    void salvar(const string& prefixo) const override {
        norm1.salvar(prefixo + "_norm1.bin");
        atencao.salvar(prefixo + "_atencao");
        norm2.salvar(prefixo + "_norm2.bin");
        oculta1.salvar(prefixo + "_oculta1.bin");
        oculta2.salvar(prefixo + "_oculta2.bin");
    }

    void carregar(const string& prefixo) override {
        norm1.carregar(prefixo + "_norm1.bin");
        atencao.carregar(prefixo + "_atencao");
        norm2.carregar(prefixo + "_norm2.bin");
        oculta1.carregar(prefixo + "_oculta1.bin");
        oculta2.carregar(prefixo + "_oculta2.bin");
    }
};