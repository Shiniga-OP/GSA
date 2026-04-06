#include "biblis/camadas/bloco.h"
#include <cassert>
#include <cstdio>

static bool aprox(float a, float b, float tol = 1e-4f) {
    return fabs(a - b) < tol;
}

void teste_prop_dim() {
    BlocoTransformer bloco(8, 4);
    vector<float> entrada(8, 0.1f);
    vector<float> saida = bloco.prop(entrada);
    assert(saida.size() == 8);
    printf("[OK] prop auto-atencao: dimensao correta\n");
}

void teste_prop_cross_dim() {
    BlocoTransformer bloco(8, 4);
    vector<float> entrada(8, 0.1f);
    vector<vector<float>> chaves = {
        vector<float>(8, 0.2f),
        vector<float>(8, 0.3f),
        vector<float>(8, 0.4f)
    };
    vector<float> saida = bloco.prop(entrada, chaves);
    assert(saida.size() == 8);
    printf("[OK] prop cross-atencao: dimensao correta\n");
}

void teste_prop_sensivel() {
    BlocoTransformer bloco(8, 4);
    vector<float> e1(8, 0.1f);
    vector<float> e2(8, 0.9f);
    vector<float> s1 = bloco.prop(e1);
    vector<float> s2 = bloco.prop(e2);
    bool diferente = false;
    for(size_t i = 0; i < 8; i++)
        if(!aprox(s1[i], s2[i])) { diferente = true; break; }
    assert(diferente);
    printf("[OK] prop: entradas diferentes produzem saidas diferentes\n");
}

void teste_retroprop_dim() {
    BlocoTransformer bloco(8, 4);
    vector<float> entrada(8, 0.1f);
    bloco.prop(entrada);
    vector<float> grad(8, 1.0f);
    auto g = bloco.retroprop(grad);
    assert(g.vetor.size() == 8);
    printf("[OK] retroprop auto-atencao: dimensao do gradEntrada correta\n");
}

void teste_retroprop_cross_dim() {
    BlocoTransformer bloco(8, 4);
    vector<float> entrada(8, 0.1f);
    vector<vector<float>> chaves = {
        vector<float>(8, 0.2f),
        vector<float>(8, 0.3f)
    };
    bloco.prop(entrada, chaves);
    vector<float> grad(8, 1.0f);
    auto g = bloco.retroprop(grad);
    assert(g.vetor.size() == 8);
    assert(g.matriz.size() == 2);
    assert(g.matriz[0].size() == 8);
    printf("[OK] retroprop cross-atencao: dimensao do gradChaves correta\n");
}

void teste_retroprop_nao_zero() {
    BlocoTransformer bloco(8, 4);
    vector<float> entrada(8, 0.5f);
    bloco.prop(entrada);
    vector<float> grad(8, 1.0f);
    auto g = bloco.retroprop(grad);
    bool temValor = false;
    for(float v : g.vetor)
        if(fabs(v) > 1e-6f) { temValor = true; break; }
    assert(temValor);
    printf("[OK] retroprop: gradiente de entrada nao e zero\n");
}

void teste_att_nao_crasha() {
    BlocoTransformer bloco(8, 4);
    vector<float> entrada(8, 0.1f);
    bloco.prop(entrada);
    vector<float> grad(8, 1.0f);
    bloco.retroprop(grad);
    bloco.att(0.01f);
    bloco.zerarGradientes();
    printf("[OK] att + zerarGradientes: sem crash\n");
}

void teste_att_atualiza() {
    BlocoTransformer bloco(8, 4);
    vector<float> entrada(8, 0.5f);
    vector<float> s1 = bloco.prop(entrada);
    vector<float> grad(8, 1.0f);
    bloco.retroprop(grad);
    bloco.att(0.1f);
    bloco.zerarGradientes();
    vector<float> s2 = bloco.prop(entrada);
    bool mudou = false;
    for(size_t i = 0; i < 8; i++)
        if(!aprox(s1[i], s2[i], 1e-3f)) { mudou = true; break; }
    assert(mudou);
    printf("[OK] att: saida muda apos atualizacao dos pesos\n");
}

void teste_num_parametros() {
    BlocoTransformer bloco(8, 4);
    assert(bloco.temParametros());
    assert(bloco.numParametros() > 0);
    printf("[OK] numParametros: %zu parametros\n", bloco.numParametros());
}

void teste_dimffn_custom() {
    BlocoTransformer bloco(8, 4, 16);
    vector<float> entrada(8, 0.1f);
    vector<float> saida = bloco.prop(entrada);
    assert(saida.size() == 8);
    printf("[OK] dimFFN customizado: prop funciona\n");
}

void teste_entrada_invalida() {
    BlocoTransformer bloco(8, 4);
    vector<float> entrada(5, 0.1f);
    bool lancou = false;
    try { bloco.prop(entrada); }
    catch(const invalid_argument&) { lancou = true; }
    assert(lancou);
    printf("[OK] entrada invalida: excecao lancada\n");
}

void teste_grad_invalido() {
    BlocoTransformer bloco(8, 4);
    vector<float> entrada(8, 0.1f);
    bloco.prop(entrada);
    vector<float> grad(5, 1.0f);
    bool lancou = false;
    try { bloco.retroprop(grad); }
    catch(const invalid_argument&) { lancou = true; }
    assert(lancou);
    printf("[OK] gradiente invalido: excecao lancada\n");
}

void teste_salvar_carregar() {
    BlocoTransformer bloco(8, 4);
    vector<float> entrada(8, 0.3f);
    vector<float> s1 = bloco.prop(entrada);
    bloco.salvar("bloco_teste");
    BlocoTransformer bloco2(8, 4);
    bloco2.carregar("bloco_teste");
    vector<float> s2 = bloco2.prop(entrada);
    for(size_t i = 0; i < 8; i++)
        assert(aprox(s1[i], s2[i], 1e-4f));
    printf("[OK] salvar/carregar: saida identica apos reload\n");
}

void teste_treino_converge() {
    BlocoTransformer bloco(8, 4);
    vector<float> entrada(8, 0.5f);
    vector<float> alvo(8, 1.0f);

    float perdaAntes = 0.0f;
    {
        vector<float> s = bloco.prop(entrada);
        for(size_t i = 0; i < 8; i++) {
            float d = s[i] - alvo[i];
            perdaAntes += d * d;
        }
    }
    for(int iter = 0; iter < 50; iter++) {
        vector<float> s = bloco.prop(entrada);
        vector<float> grad(8);
        for(size_t i = 0; i < 8; i++) grad[i] = 2.0f * (s[i] - alvo[i]);
        bloco.retroprop(grad);
        bloco.att(0.01f);
        bloco.zerarGradientes();
    }
    float perdaDepois = 0.0f;
    {
        vector<float> s = bloco.prop(entrada);
        for(size_t i = 0; i < 8; i++) {
            float d = s[i] - alvo[i];
            perdaDepois += d * d;
        }
    }
    assert(perdaDepois < perdaAntes);
    printf("[OK] treino: perda diminui apos 50 iteracoes (%.4f -> %.4f)\n",
        perdaAntes, perdaDepois);
}

int main() {
    printf("=== BlocoTransformer ===\n");
    teste_prop_dim();
    teste_prop_cross_dim();
    teste_prop_sensivel();
    teste_retroprop_dim();
    teste_retroprop_cross_dim();
    teste_retroprop_nao_zero();
    teste_att_nao_crasha();
    teste_att_atualiza();
    teste_num_parametros();
    teste_dimffn_custom();
    teste_entrada_invalida();
    teste_grad_invalido();
    teste_salvar_carregar();
    teste_treino_converge();
    printf("=== todos os testes passaram ===\n");
    return 0;
}