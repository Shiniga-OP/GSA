// teste_conversacional.cpp
#include "biblis/modelo.h"
#include <cassert>
#include <cstdio>
#include <cstring>
#include <string>

// === dados de treino ===
static vector<string> corpusTeste() {
    return {
        "o gato comeu o rato",
        "o rato fugiu do gato",
        "o cachorro latiu alto",
        "a casa e grande e bonita",
        "o sol nasceu cedo hoje",
        "ela gosta de cafe pela manha",
        "o livro esta na mesa",
        "o menino correu no parque",
        "você me observou de longe",
        "eu estou aprendendo"
    };
}

// === testes ===
void teste_construcao() {
    auto corpus = corpusTeste();
    TreinadorBPE treinador; treinador.treinar(corpus, 100);
    TokenizadorBPE tok(treinador.merges); tok.construirVocab(corpus);
    Modelo modelo(tok, 16, 8, 2, 64, 0, "relu", 1e-3f, 1, 0);
    assert(modelo.net.numParametros() > 0);
    printf("[OK] construcao: %zu parametros\n", modelo.net.numParametros());
}

void teste_treino_nao_crasha() {
    auto corpus = corpusTeste();
    TreinadorBPE treinador; treinador.treinar(corpus, 100);
    TokenizadorBPE tok(treinador.merges); tok.construirVocab(corpus);
    Modelo modelo(tok, 16, 8, 2, 64, 0, "relu", 1e-3f, 1, 0);
    modelo.treinar(corpus);
    printf("[OK] treino 1 epoca: sem crash\n");
}

void teste_treino_converge() {
    auto corpus = corpusTeste();
    TreinadorBPE treinador; treinador.treinar(corpus, 200);
    TokenizadorBPE tok(treinador.merges); tok.construirVocab(corpus);

    float perdaAntes = 0.0f, perdaDepois = 0.0f;
    Modelo modelo(tok, 16, 8, 2, 64, 0, "relu", 1e-2f, 1, 0);
    {
        auto& seq0 = corpus[0];
        vector<int> ids = tok.codificar(seq0);
        ids.insert(ids.begin(), 0); ids.push_back(2);
        vector<int> entrada(ids.begin(), ids.end()-1);
        vector<int> alvo(ids.begin()+1, ids.end());
        vector<float> x = modelo.net.embedding.prop((size_t)entrada[0]);
        x = modelo.net.posicional.prop(x, 0);
        for (auto& b : modelo.net.blocos) x = b->prop(x);
        vector<float> logits = modelo.net.projecao.prop(x);
        CamadaPerda cp; perdaAntes = cp.prop(logits, (size_t)alvo[0]);
    }
    Modelo modelo2(tok, 16, 8, 2, 64, 0, "relu", 1e-2f, 10, 0);
    modelo2.treinar(corpus);
    {
        auto& seq0 = corpus[0];
        vector<int> ids = tok.codificar(seq0);
        ids.insert(ids.begin(), 0); ids.push_back(2);
        vector<int> entrada(ids.begin(), ids.end()-1);
        vector<int> alvo(ids.begin()+1, ids.end());
        vector<float> x = modelo2.net.embedding.prop((size_t)entrada[0]);
        x = modelo2.net.posicional.prop(x, 0);
        for (auto& b : modelo2.net.blocos) x = b->prop(x);
        vector<float> logits = modelo2.net.projecao.prop(x);
        CamadaPerda cp; perdaDepois = cp.prop(logits, (size_t)alvo[0]);
    }
    assert(perdaDepois < perdaAntes);
    printf("[OK] treino converge: perda %.4f -> %.4f\n", perdaAntes, perdaDepois);
}

void teste_geracao_nao_vazia() {
    auto corpus = corpusTeste();
    TreinadorBPE treinador; treinador.treinar(corpus, 100);
    TokenizadorBPE tok(treinador.merges); tok.construirVocab(corpus);
    Modelo modelo(tok, 16, 8, 2, 64, 0, "relu", 1e-3f, 2, 0);
    modelo.treinar(corpus);
    string saida = modelo.gerar("o gato", 16, 0.8f);
    assert(!saida.empty());
    printf("[OK] geracao: \"%s\"\n", saida.c_str());
}

void teste_geracao_deterministica() {
    auto corpus = corpusTeste();
    TreinadorBPE treinador; treinador.treinar(corpus, 100);
    TokenizadorBPE tok(treinador.merges); tok.construirVocab(corpus);
    Modelo modelo(tok, 16, 8, 2, 64, 0, "relu", 1e-3f, 2, 0);
    modelo.treinar(corpus);
    string s1 = modelo.gerar("o gato", 16, 0.0f);
    string s2 = modelo.gerar("o gato", 16, 0.0f);
    assert(s1 == s2);
    printf("[OK] geracao deterministica: \"%s\"\n", s1.c_str());
}

void teste_salvar_carregar() {
    auto corpus = corpusTeste();
    TreinadorBPE treinador; treinador.treinar(corpus, 100);
    TokenizadorBPE tok(treinador.merges); tok.construirVocab(corpus);
    Modelo modelo(tok, 16, 8, 2, 64, 0, "relu", 1e-3f, 2, 0);
    modelo.treinar(corpus);
    string s1 = modelo.gerar("o gato", 8, 0.0f);
    modelo.salvar("modelo_teste");
    Modelo modelo2(tok, 16, 8, 2, 64, 0, "relu", 1e-3f, 1, 0);
    modelo2.carregar("modelo_teste");
    string s2 = modelo2.gerar("o gato", 8, 0.0f);
    assert(s1 == s2);
    printf("[OK] salvar/carregar: saida identica apos reload\n");
}

void teste_corpus_vazio() {
    auto corpus = corpusTeste();
    TreinadorBPE treinador; treinador.treinar(corpus, 100);
    TokenizadorBPE tok(treinador.merges); tok.construirVocab(corpus);
    Modelo modelo(tok, 16, 8, 2, 64, 0, "relu", 1e-3f, 1, 0);
    modelo.treinar({});
    printf("[OK] corpus vazio: sem crash\n");
}

void teste_prompt_vazio() {
    auto corpus = corpusTeste();
    TreinadorBPE treinador; treinador.treinar(corpus, 100);
    TokenizadorBPE tok(treinador.merges); tok.construirVocab(corpus);
    Modelo modelo(tok, 16, 8, 2, 64, 0, "relu", 1e-3f, 2, 0);
    modelo.treinar(corpus);
    string saida = modelo.gerar("", 8, 0.8f);
    printf("[OK] comando vazio: sem crash, saida=\"%s\"\n", saida.c_str());
}

void teste_prompt_longo() {
    auto corpus = corpusTeste();
    TreinadorBPE treinador; treinador.treinar(corpus, 100);
    TokenizadorBPE tok(treinador.merges); tok.construirVocab(corpus);
    Modelo modelo(tok, 16, 8, 2, 64, 0, "relu", 1e-3f, 2, 0);
    modelo.treinar(corpus);
    string comando = "o gato comeu o rato e o cachorro latiu muito forte la fora hoje cedo pela manha";
    string saida = modelo.gerar(comando, 8, 0.0f);
    assert(!saida.empty());
    printf("[OK] comando longo (>ctx): sem crash, saida=\"%s\"\n", saida.c_str());
}

int main(int argc, char* argv[]) {
    printf("=== Modelo ===\n");
    teste_construcao();
    teste_treino_nao_crasha();
    teste_treino_converge();
    teste_geracao_nao_vazia();
    teste_geracao_deterministica();
    teste_salvar_carregar();
    teste_corpus_vazio();
    teste_prompt_vazio();
    teste_prompt_longo();
    printf("=== todos os testes passaram ===\n");
    return 0;
}
