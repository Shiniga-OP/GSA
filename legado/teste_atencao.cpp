// teste_atencao.cpp
#include <iostream>
#include <vector>
#include <cassert>
#include <fstream>
#include "biblis/ativas.h"
#include "biblis/util.h"
#include "biblis/otimizadores.h"
#include "biblis/camadas/atencao.h"

using namespace std;

int totalTestes = 0;
int testesPassados = 0;

void verificar(bool cond, const string& desc) {
    totalTestes++;
    if(cond) { testesPassados++; cout << "  [OK] " << desc << endl; }
    else cout << "  [FALHOU] " << desc << endl;
}
void verificarPerto(float a, float b, const string& desc, float tol = 1e-4f) {
    verificar(abs(a-b) < tol, desc + " (" + to_string(a) + " ~= " + to_string(b) + ")");
}
void secao(const string& t) { cout << "\n=== " << t << " ===" << endl; }

// =====================================================================
// TESTE 1: saída tem dimensão correta
// =====================================================================
void testeDimensoes() {
    secao("Dimensões de saída");

    CamadaAtencao at(8, 4, 6); // dim=8, dimAtencao=4, dimSaida=6

    vector<float> estado(8, 0.5f);
    vector<vector<float>> chaves = {
        vector<float>(8, 0.1f),
        vector<float>(8, 0.2f),
        vector<float>(8, 0.3f)
    };

    auto saida = at.prop(estado, chaves);
    verificar(saida.size() == 6, "saída tem dimSaida=6 elementos");
    verificar(at.pesosAtencao().size() == 3, "pesos de atenção têm m=3 elementos");

    // pesos devem somar 1 (softmax)
    float soma = 0.0f;
    for(float p : at.pesosAtencao()) soma += p;
    verificarPerto(soma, 1.0f, "pesos de atenção somam 1");

    // todos os pesos devem ser positivos
    bool todosPositivos = true;
    for(float p : at.pesosAtencao()) if(p <= 0.0f) todosPositivos = false;
    verificar(todosPositivos, "todos os pesos de atenção são positivos");
}

// =====================================================================
// TESTE 2: atenção foca na chave mais similar à consulta
// =====================================================================
void testeFoco() {
    secao("Foco — chave similar à consulta recebe peso maior");

    // dimensão pequena pra controlar manualmente
    // Pq = identidade, Pk = identidade (pra poder prever o comportamento)
    CamadaAtencao at(4, 4, 4);

    // força Pq e Pk pra identidade pra tornar o teste determinístico
    at.Pq = identidade(4);
    at.Pk = identidade(4);
    at.Pv = identidade(4);

    // consulta/estado = [1,0,0,0]
    vector<float> estado = {1.0f, 0.0f, 0.0f, 0.0f};

    // 3 chaves: a primeira é idêntica à consulta, as outras são ortogonais
    vector<vector<float>> chaves = {
        {1.0f, 0.0f, 0.0f, 0.0f}, // similar: dot = 1.0
        {0.0f, 1.0f, 0.0f, 0.0f}, // ortogonal: dot = 0.0
        {0.0f, 0.0f, 1.0f, 0.0f}  // ortogonal: dot = 0.0
    };

    at.prop(estado, chaves);
    auto& pesos = at.pesosAtencao();

    verificar(pesos[0] > pesos[1], "chave similar [0] tem peso maior que ortogonal [1]");
    verificar(pesos[0] > pesos[2], "chave similar [0] tem peso maior que ortogonal [2]");
    verificarPerto(pesos[1], pesos[2], "chaves ortogonais têm pesos iguais entre si");

    cout << "  pesos: [" << pesos[0] << ", " << pesos[1] << ", " << pesos[2] << "]" << endl;
}

// =====================================================================
// TESTE 3: verificação numérica de gradientes (diferenças finitas)
// =====================================================================
float perdaSimples(CamadaAtencao& at,
                   const vector<float>& estado,
                   const vector<vector<float>>& chaves,
                   const vector<float>& alvo) {
    auto saida = at.prop(estado, chaves);
    float loss = 0.0f;
    for(size_t i = 0; i < saida.size(); i++) {
        float d = saida[i] - alvo[i];
        loss += d * d;
    }
    return loss * 0.5f;
}

void testeGradientesNumericos() {
    secao("Gradientes numéricos — Pq, Pk, Pv");

    const float h = 1e-4f;
    const float tol = 1e-2f; // tolerância de 1% pra gradientes numéricos

    CamadaAtencao at(3, 3, 3);
    // pesos fixos pra reprodutibilidade
    at.Pq = {{0.1f,-0.2f,0.3f},{0.4f,0.1f,-0.1f},{-0.2f,0.3f,0.2f}};
    at.Pk = {{0.2f, 0.1f,-0.3f},{-0.1f,0.4f,0.2f},{0.3f,-0.1f,0.1f}};
    at.Pv = {{0.3f,-0.1f,0.2f},{0.1f,0.2f,-0.3f},{-0.2f,0.1f,0.4f}};

    vector<float> estado  = {0.5f, -0.3f, 0.8f};
    vector<vector<float>> chaves = {
        { 0.2f,  0.7f, -0.1f},
        {-0.4f,  0.3f,  0.6f}
    };
    vector<float> alvo = {1.0f, 0.0f, -1.0f};

    // forward + backward
    auto saida = at.prop(estado, chaves);
    vector<float> gradSaida(3);
    for(size_t i = 0; i < 3; i++) gradSaida[i] = saida[i] - alvo[i]; // dL/dsaida = saida - alvo

    at.zerarGradientes();
    at.retroprop(gradSaida);

    // verifica alguns elementos de gradPq via diferenças finitas
    int verificados = 0;
    for(size_t i = 0; i < 3 && verificados < 3; i++) {
        for(size_t j = 0; j < 3 && verificados < 3; j++) {
            float orig = at.Pq[i][j];

            at.Pq[i][j] = orig + h;
            float lossP = perdaSimples(at, estado, chaves, alvo);

            at.Pq[i][j] = orig - h;
            float lossM = perdaSimples(at, estado, chaves, alvo);

            at.Pq[i][j] = orig;

            float gradNum = (lossP - lossM) / (2.0f * h);
            float gradAna = at.gradPq[i][j];

            string desc = "gradPq[" + to_string(i) + "][" + to_string(j) + 
                          "] num=" + to_string(gradNum) + " ana=" + to_string(gradAna);
            verificar(abs(gradNum - gradAna) < tol, desc);
            verificados++;
        }
    }

    // verifica alguns de gradPv
    verificados = 0;
    for(size_t i = 0; i < 3 && verificados < 3; i++) {
        for(size_t j = 0; j < 3 && verificados < 3; j++) {
            float orig = at.Pv[i][j];

            at.Pv[i][j] = orig + h;
            float lossP = perdaSimples(at, estado, chaves, alvo);

            at.Pv[i][j] = orig - h;
            float lossM = perdaSimples(at, estado, chaves, alvo);

            at.Pv[i][j] = orig;

            float gradNum = (lossP - lossM) / (2.0f * h);
            float gradAna = at.gradPv[i][j];

            string desc = "gradPv[" + to_string(i) + "][" + to_string(j) + 
                          "] num=" + to_string(gradNum) + " ana=" + to_string(gradAna);
            verificar(abs(gradNum - gradAna) < tol, desc);
            verificados++;
        }
    }

    // verifica alguns de gradPk
    verificados = 0;
    for(size_t i = 0; i < 3 && verificados < 3; i++) {
        for(size_t j = 0; j < 3 && verificados < 3; j++) {
            float orig = at.Pk[i][j];

            at.Pk[i][j] = orig + h;
            float lossP = perdaSimples(at, estado, chaves, alvo);

            at.Pk[i][j] = orig - h;
            float lossM = perdaSimples(at, estado, chaves, alvo);

            at.Pk[i][j] = orig;

            float gradNum = (lossP - lossM) / (2.0f * h);
            float gradAna = at.gradPk[i][j];

            string desc = "gradPk[" + to_string(i) + "][" + to_string(j) + 
                          "] num=" + to_string(gradNum) + " ana=" + to_string(gradAna);
            verificar(abs(gradNum - gradAna) < tol, desc);
            verificados++;
        }
    }
}

// =====================================================================
// TESTE 4: gradiente pro estado de entrada
// =====================================================================
void testeGradEstado() {
    secao("Gradiente pro estado de entrada");

    const float h = 1e-4f;
    const float tol = 1e-2f;

    CamadaAtencao at(3, 3, 3);
    at.Pq = {{0.5f, 0.1f,-0.2f},{-0.1f,0.4f,0.3f},{0.2f,-0.3f,0.1f}};
    at.Pk = {{0.3f,-0.2f,0.1f},{ 0.1f,0.2f,0.4f},{-0.3f,0.1f,0.2f}};
    at.Pv = {{0.1f, 0.4f,-0.1f},{-0.2f,0.1f,0.3f},{ 0.3f,-0.1f,0.2f}};

    vector<float> estado = {0.3f, -0.5f, 0.7f};
    vector<vector<float>> chaves = {
        {0.1f, 0.6f, -0.2f},
        {-0.3f, 0.4f, 0.5f}
    };
    vector<float> alvo = {0.5f, -0.5f, 0.5f};

    auto saida = at.prop(estado, chaves);
    vector<float> gradSaida(3);
    for(size_t i = 0; i < 3; i++) gradSaida[i] = saida[i] - alvo[i];

    at.zerarGradientes();
    auto grad = at.retroprop(gradSaida);

    for(size_t j = 0; j < 3; j++) {
        float orig = estado[j];

        estado[j] = orig + h;
        float lossP = perdaSimples(at, estado, chaves, alvo);

        estado[j] = orig - h;
        float lossM = perdaSimples(at, estado, chaves, alvo);

        estado[j] = orig;

        float gradNum = (lossP - lossM) / (2.0f * h);
        float gradAna = grad.gradEstado[j];

        string desc = "gradEstado[" + to_string(j) + "] num=" + 
                      to_string(gradNum) + " ana=" + to_string(gradAna);
        verificar(abs(gradNum - gradAna) < tol, desc);
    }
}

// =====================================================================
// TESTE 5: treino — aprende a recuperar o valor associado à chave correta
// =====================================================================
void testeTreino() {
    secao("Treino — recuperação de memória simples");

    // tarefa: dado estado (consulta), a atenção deve buscar a entrada de memória
    // com chave mais similar e retornar o valor associado
    //
    // CHAVES e VALORES são diferentes entre si pra eliminar ambiguidade:
    // se chave[i] == valor[i], Pv pode aprender qualquer permutação e
    // ainda minimizar MSE sem que a atenção foque no índice certo.
    //
    // aqui: chaves são vetores de busca (one-hot), valores são alvos distintos
    // a entrada de memória é [chave|valor] concatenados em dim=8
    // estado (consulta) = só a parte da chave, padded com zeros

    // O teste correto de foco:
    // - consultas têm informação APENAS nos primeiros 4 elementos
    // - memória tem chaves nos primeiros 4 e valores únicos nos últimos 4
    // - Pk deve aprender a comparar pelos primeiros 4 (chaves)
    // - Pv deve aprender a extrair os últimos 4 (valores)
    // - como as consultas têm zeros nos últimos 4, Pv não pode "trapacear"
    //   usando a consulta pra reconstruir o valor diretamente
    //
    // adicionalmente: chaves na memória e consultas NÃO são idênticas
    // (consulta tem ruído diferente) → a única solução é focar no índice certo
    const size_t D = 8;

    // memória: [chave(4) | valor(4)] — valores são distintos e ortogonais
    vector<vector<float>> memoria = {
        {1.0f, 0.2f, 0.0f, 0.0f,   0.9f, 0.1f, 0.0f, 0.0f}, // entrada 0
        {0.0f, 1.0f, 0.3f, 0.0f,   0.0f, 0.8f, 0.2f, 0.0f}, // entrada 1
        {0.1f, 0.0f, 1.0f, 0.2f,   0.0f, 0.0f, 0.9f, 0.1f}, // entrada 2
    };
    // consultas: parte da chave com pequena variação, parte do valor = zero
    // não é idêntica à chave da memória → o modelo é forçado a aprender similaridade
    vector<vector<float>> consultas = {
        {0.9f, 0.3f, 0.0f, 0.0f,  0,0,0,0}, // similar à entrada 0
        {0.0f, 0.9f, 0.4f, 0.0f,  0,0,0,0}, // similar à entrada 1
        {0.2f, 0.0f, 0.9f, 0.3f,  0,0,0,0}, // similar à entrada 2
    };
    // alvos: a parte do valor da entrada correspondente
    vector<vector<float>> alvos = {
        {0.9f, 0.1f, 0.0f, 0.0f},
        {0.0f, 0.8f, 0.2f, 0.0f},
        {0.0f, 0.0f, 0.9f, 0.1f},
    };

    // dimSaida=4 (só o valor), dim=8 (chave+valor concatenados)
    CamadaAtencao at(D, D, 4);
    at.defOtimizadores(
        make_unique<Adam>(0.01f),
        make_unique<Adam>(0.01f),
        make_unique<Adam>(0.01f)
    );

    float ultimoErro = 999.0f;
    for(int epoca = 0; epoca < 2000; epoca++) {
        float erroTotal = 0.0f;

        for(size_t ex = 0; ex < consultas.size(); ex++) {
            at.zerarGradientes();

            auto saida = at.prop(consultas[ex], memoria);

            vector<float> gradSaida(4);
            float erro = 0.0f;
            for(size_t i = 0; i < 4; i++) {
                float d = saida[i] - alvos[ex][i];
                gradSaida[i] = d;
                erro += d * d;
            }
            erroTotal += erro * 0.5f;

            at.retroprop(gradSaida);
            at.att(0.01f);
        }
        ultimoErro = erroTotal / consultas.size();
    }

    verificar(ultimoErro < 0.01f, "erro < 0.01 após 2000 épocas (" + to_string(ultimoErro) + ")");

    // o que importa: dado a consulta correta, a saída corresponde ao valor correto
    // o índice interno do foco é irrelevante — pode ser qualquer permutação válida
    for(size_t ex = 0; ex < consultas.size(); ex++) {
        auto saida = at.prop(consultas[ex], memoria);

        // saída deve ser próxima do alvo correspondente
        float erroEx = 0.0f;
        for(size_t i = 0; i < 4; i++) {
            float d = saida[i] - alvos[ex][i];
            erroEx += d * d;
        }
        verificar(erroEx < 0.01f,
            "consulta " + to_string(ex) + ": saída correta (erro=" + to_string(erroEx) + ")");
    }

    // saídas de consultas diferentes devem ser distintas entre si
    auto s0 = at.prop(consultas[0], memoria);
    auto s1 = at.prop(consultas[1], memoria);
    auto s2 = at.prop(consultas[2], memoria);
    float dist01 = 0, dist02 = 0, dist12 = 0;
    for(size_t i = 0; i < 4; i++) {
        dist01 += (s0[i]-s1[i])*(s0[i]-s1[i]);
        dist02 += (s0[i]-s2[i])*(s0[i]-s2[i]);
        dist12 += (s1[i]-s2[i])*(s1[i]-s2[i]);
    }
    verificar(dist01 > 0.1f, "saídas de consultas 0 e 1 são distintas (dist=" + to_string(dist01) + ")");
    verificar(dist02 > 0.1f, "saídas de consultas 0 e 2 são distintas (dist=" + to_string(dist02) + ")");
    verificar(dist12 > 0.1f, "saídas de consultas 1 e 2 são distintas (dist=" + to_string(dist12) + ")");
}

// =====================================================================
// TESTE 6: zerarGradientes realmente zera tudo
// =====================================================================
void testeZerar() {
    secao("zerarGradientes");

    CamadaAtencao at(3, 3, 3);
    vector<float> estado = {1.0f, 0.0f, 0.0f};
    vector<vector<float>> chaves = {{1.0f,0.0f,0.0f},{0.0f,1.0f,0.0f}};
    vector<float> gradSaida = {1.0f, 1.0f, 1.0f};

    at.prop(estado, chaves);
    at.retroprop(gradSaida);

    // tem gradientes agora
    float somaAntes = 0.0f;
    for(auto& l : at.gradPq) for(float g : l) somaAntes += abs(g);
    verificar(somaAntes > 0.0f, "gradientes não-zero antes de zerar");

    at.zerarGradientes();

    float somaDepois = 0.0f;
    for(auto& l : at.gradPq) for(float g : l) somaDepois += abs(g);
    for(auto& l : at.gradPk) for(float g : l) somaDepois += abs(g);
    for(auto& l : at.gradPv) for(float g : l) somaDepois += abs(g);
    verificarPerto(somaDepois, 0.0f, "todos os gradientes zerados após zerarGradientes");
}

// =====================================================================
// MAIN
// =====================================================================
int main() {
    cout << "=====================================================" << endl;
    cout << "  TESTES — CamadaAtencao" << endl;
    cout << "=====================================================" << endl;

    testeDimensoes();
    testeFoco();
    testeGradientesNumericos();
    testeGradEstado();
    testeTreino();
    testeZerar();

    cout << "\n=====================================================" << endl;
    cout << "  RESULTADO: " << testesPassados << "/" << totalTestes << " testes passaram" << endl;
    if(testesPassados == totalTestes) cout << "  TUDO OK" << endl;
    else cout << "  " << (totalTestes - testesPassados) << " FALHARAM" << endl;
    cout << "=====================================================" << endl;

    return (testesPassados == totalTestes) ? 0 : 1;
}