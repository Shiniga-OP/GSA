#include "biblis/camadas/posicional.h"
#include <cassert>
#include <cstdio>
#include <cmath>

static bool aprox(float a, float b, float tol = 1e-5f) {
    return fabs(a - b) < tol;
}

// 1. sinusoidal: PE[0] deve ser all-sin(0)/cos(0) = 0/1 alternado
void teste_sinusoidal_pos0() {
    CamadaPosicional cp(8, 64, false);
    for(size_t i = 0; i < 4; i++) {
        assert(aprox(cp.PE[0][2*i],     0.0f)); // sin(0) = 0
        assert(aprox(cp.PE[0][2*i + 1], 1.0f)); // cos(0) = 1
    }
    printf("[OK] sinusoidal pos=0: sin=0, cos=1\n");
}

// 2. sinusoidal: prop soma corretamente
void teste_sinusoidal_prop() {
    CamadaPosicional cp(4, 64, false);
    vector<float> entrada = {1.0f, 2.0f, 3.0f, 4.0f};
    vector<float> saida = cp.prop(entrada, 1);
    for(size_t i = 0; i < 4; i++) {
        assert(aprox(saida[i], entrada[i] + cp.PE[1][i]));
    }
    printf("[OK] sinusoidal prop: saida = entrada + PE[pos]\n");
}

// 3. sinusoidal: retroprop passa gradiente intacto (sem acumular nada)
void teste_sinusoidal_retroprop() {
    CamadaPosicional cp(4, 64, false);
    vector<float> entrada = {1.0f, 0.5f, -1.0f, 2.0f};
    cp.prop(entrada, 3);
    vector<float> grad = {0.1f, 0.2f, 0.3f, 0.4f};
    auto g = cp.retroprop(grad);
    for(size_t i = 0; i < 4; i++) {
        assert(aprox(g.vetor[i], grad[i]));
    }
    printf("[OK] sinusoidal retroprop: gradiente passa intacto\n");
}

// 4. sinusoidal: att e zerarGradientes são no-op (sem crash)
void teste_sinusoidal_att_noop() {
    CamadaPosicional cp(4, 64, false);
    cp.zerarGradientes();
    cp.att(0.01f);
    assert(!cp.temParametros());
    assert(cp.numParametros() == 0);
    printf("[OK] sinusoidal att/zerarGradientes: no-op sem crash\n");
}

// 5. treinável: prop soma PE aprendido
void teste_treinavel_prop() {
    CamadaPosicional cp(4, 16, true);
    // zera PE pra resultado previsível
    for(auto& l : cp.PE) fill(l.begin(), l.end(), 0.0f);
    vector<float> entrada = {1.0f, 2.0f, 3.0f, 4.0f};
    vector<float> saida = cp.prop(entrada, 2);
    for(size_t i = 0; i < 4; i++) {
        assert(aprox(saida[i], entrada[i])); // PE=0, saida=entrada
    }
    printf("[OK] treinavel prop: saida = entrada + PE[pos]\n");
}

// 6. treinável: retroprop acumula gradiente em gradE[pos]
void teste_treinavel_retroprop() {
    CamadaPosicional cp(4, 16, true);
    vector<float> entrada = {1.0f, 1.0f, 1.0f, 1.0f};
    cp.prop(entrada, 5);
    vector<float> grad = {0.1f, 0.2f, 0.3f, 0.4f};
    auto g = cp.retroprop(grad);
    // gradiente da entrada passa intacto
    for(size_t i = 0; i < 4; i++) assert(aprox(g.vetor[i], grad[i]));
    // acumulado em gradE[5]
    for(size_t i = 0; i < 4; i++) assert(aprox(cp.gradE[5][i], grad[i]));
    printf("[OK] treinavel retroprop: grad acumulado em gradE[pos]\n");
}

// 7. treinável: att SGD manual atualiza PE
void teste_treinavel_att() {
    CamadaPosicional cp(4, 16, true);
    for(auto& l : cp.PE)   fill(l.begin(), l.end(), 1.0f);
    for(auto& l : cp.gradE) fill(l.begin(), l.end(), 0.5f);
    cp.att(0.1f);
    for(size_t i = 0; i < 4; i++) {
        assert(aprox(cp.PE[3][i], 1.0f - 0.1f * 0.5f)); // 1 - 0.05 = 0.95
    }
    printf("[OK] treinavel att: PE atualizado corretamente\n");
}

// 8. treinável: zerarGradientes zera gradE
void teste_treinavel_zerar() {
    CamadaPosicional cp(4, 16, true);
    for(auto& l : cp.gradE) fill(l.begin(), l.end(), 9.9f);
    cp.zerarGradientes();
    for(const auto& l : cp.gradE)
        for(float v : l) assert(aprox(v, 0.0f));
    printf("[OK] treinavel zerarGradientes: gradE zerado\n");
}

// 9. salvar/carregar sinusoidal: PE recalculado igual
void teste_salvar_carregar_sinusoidal() {
    CamadaPosicional cp(8, 32, false);
    cp.salvar("pos_sin.bin");

    CamadaPosicional cp2(1, 1, false); // dims erradas intencionalmente
    cp2.carregar("pos_sin.bin");

    for(size_t p = 0; p < 32; p++)
        for(size_t i = 0; i < 8; i++)
            assert(aprox(cp.PE[p][i], cp2.PE[p][i]));
    printf("[OK] salvar/carregar sinusoidal: PE idêntico\n");
}

// 10. salvar/carregar treinável: PE preservado
void teste_salvar_carregar_treinavel() {
    CamadaPosicional cp(6, 10, true);
    // força valores conhecidos
    for(size_t i = 0; i < 10; i++)
        for(size_t j = 0; j < 6; j++)
            cp.PE[i][j] = (float)(i * 6 + j) * 0.01f;

    cp.salvar("pos_tre.bin");

    CamadaPosicional cp2(1, 1, true);
    cp2.carregar("pos_tre.bin");

    for(size_t i = 0; i < 10; i++)
        for(size_t j = 0; j < 6; j++)
            assert(aprox(cp.PE[i][j], cp2.PE[i][j]));
    printf("[OK] salvar/carregar treinavel: PE preservado\n");
}

// 11. dim ímpar: não acessa fora do vetor
void teste_dim_impar() {
    CamadaPosicional cp(5, 8, false);
    vector<float> entrada(5, 1.0f);
    vector<float> saida = cp.prop(entrada, 0);
    assert(saida.size() == 5);
    printf("[OK] dim impar: sem acesso fora do vetor\n");
}

// 12. posição fora do limite lança exceção
void teste_pos_invalida() {
    CamadaPosicional cp(4, 8, false);
    vector<float> entrada(4, 0.0f);
    bool lancou = false;
    try { cp.prop(entrada, 8); }
    catch(const invalid_argument&) { lancou = true; }
    assert(lancou);
    printf("[OK] posição inválida: exceção lançada\n");
}

int main() {
    printf("=== CamadaPosicional ===\n");
    teste_sinusoidal_pos0();
    teste_sinusoidal_prop();
    teste_sinusoidal_retroprop();
    teste_sinusoidal_att_noop();
    teste_treinavel_prop();
    teste_treinavel_retroprop();
    teste_treinavel_att();
    teste_treinavel_zerar();
    teste_salvar_carregar_sinusoidal();
    teste_salvar_carregar_treinavel();
    teste_dim_impar();
    teste_pos_invalida();
    printf("=== todos os testes passaram ===\n");
    return 0;
}