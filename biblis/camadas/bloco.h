// biblis/camadas/bloco.h
#pragma once
#include "camada.h"
#include "atencao.h"
#include "densa.h"
#include "norm.h"

// bloco transformer padrão com conexões residuais

// fluxo:
// x -> norm1 -> atencao(x, chaves) -> + x  -> x1
// x1 -> norm2 -> oculta -> + x1 -> saida

// oculta: Densa(dim -> dimoculta, ativacao) -> Densa(dimoculta -> dim, linear)
// dimoculta padrão = 4 * dim(convencional)

// chaves externas: se nullptr, usa a propria entrada como chaves(auto-atenção)

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

    // cache pra retropropagação
    vector<float> entradaCache; // x original
    vector<float> x1Cache; // após residual da atenção
    vector<float> normSaida1; // saída da norm1
    vector<float> normSaida2; // saída da norm2
    vector<float> atencaoSaida; // saída da atenção(antes do residual)
    vector<float> oculta1Saida; // saída do oculta1
    vector<vector<float>> chavesCache; // chaves usadas(pra retroprop)

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
          norm1(dim,  1e-5f, nome + "_norm1"),
          atencao(dim, dimAtencao, dim, nome + "_atencao"),
          norm2(dim,  1e-5f, nome + "_norm2"),
          oculta1(dim,  (dimoculta > 0 ? dimoculta : 4 * dim), ativacaooculta, true, nome + "_oculta1"),
          oculta2((dimoculta > 0 ? dimoculta : 4 * dim), dim, "linear", true, nome + "_oculta2") {
        tipo = "BlocoTransformer";
    }

    // prop sem chaves externas: auto-atenção(x atende a si mesmo)
    vector<float> prop(const vector<float>& entrada) override {
        // monta chaves = {entrada} atenção sobre si mesmo
        return prop(entrada, {entrada});
    }

    // prop com chaves externas: cross-atenção
    vector<float> prop(
        const vector<float>& entrada,
        const vector<vector<float>>& chaves
    ) {
        if(entrada.size() != dim)
            throw invalid_argument("[" + nome + "]: dimensão de entrada incorreta");

        entradaCache = entrada;
        chavesCache  = chaves;

        // === bloco de atenção ===
        normSaida1 = norm1.prop(entrada);
        atencaoSaida = atencao.prop(normSaida1, chaves);

        // residual
        x1Cache.resize(dim);
        for(size_t i = 0; i < dim; i++)
            x1Cache[i] = entrada[i] + atencaoSaida[i];

        // === bloco oculta ===
        normSaida2 = norm2.prop(x1Cache);
        oculta1Saida  = oculta1.prop(normSaida2);
        vector<float> oculta2Saida = oculta2.prop(oculta1Saida);

        // residual
        vector<float> saida(dim);
        for(size_t i = 0; i < dim; i++) {
            saida[i] = x1Cache[i] + oculta2Saida[i];
        }
        return saida;
    }

    // retroprop: gradSaida(dim,) ->
    // retorna GradGenerico com:
    // vetor = gradEntrada(dim,)
    // matriz = gradChaves(m x dim)
    GradGenerico retroprop(const vector<float>& gradSaida) override {
        if(gradSaida.size() != dim) {
            throw invalid_argument("[" + nome + "]: dimensão do gradiente incorreta");
        }
        // === residual oculta: grad flui pra x1 e pra oculta2 ===
        // saida = x1 + oculta2Saida => dL/dx1 += gradSaida, dL/doculta2 = gradSaida
        vector<float> gradX1(gradSaida); // copia pra acumular residual

        auto goculta2 = oculta2.retroprop(gradSaida);
        auto goculta1 = oculta1.retroprop(goculta2.vetor);
        auto gNorm2 = norm2.retroprop(goculta1.vetor);

        for(size_t i = 0; i < dim; i++) {
            gradX1[i] += gNorm2.vetor[i];
        }
        // === residual atenção: grad flui pra entrada e pra atencao ===
        // x1 = entrada + atencaoSaida  =>  dL/dentrada += gradX1, dL/datencao = gradX1
        vector<float> gradEntrada(gradX1); // copia pra acumular residual

        auto gAtencao = atencao.retroprop(gradX1);
        auto gNorm1   = norm1.retroprop(gAtencao.gradEstado);

        for(size_t i = 0; i < dim; i++) {
            gradEntrada[i] += gNorm1.vetor[i];
        }
        return GradGenerico(gradEntrada, gAtencao.gradChaves);
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

    // propaga otimizador pra todas as sub-camadas
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
        // atencao tem otimizadores separados por projeção
        // usa defOtimizador base(aplica so a um; pra controle fino use defOtimizadores da atencao)
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