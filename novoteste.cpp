// teste.cpp
// compila: g++ -std=c++17 -O2 -o teste teste.cpp
#include <iostream>
#include <vector>
#include <cmath>
#include <cassert>
#include "biblis/ativas.h"
#include "biblis/util.h"
#include "biblis/otimizadores.h"
#include "biblis/camadas.h"

using namespace std;

// =====================================================================
// UTILITÁRIOS DE TESTE
// =====================================================================

int totalTestes = 0;
int testesPassados = 0;

void verificar(bool condicao, const string& descricao) {
    totalTestes++;
    if(condicao) {
        testesPassados++;
        cout << "  [OK] " << descricao << endl;
    } else {
        cout << "  [FALHOU] " << descricao << endl;
    }
}

void verificarPerto(float a, float b, const string& descricao, float tolerancia = 1e-4f) {
    verificar(abs(a - b) < tolerancia, descricao + " (" + to_string(a) + " ~= " + to_string(b) + ")");
}

void secao(const string& titulo) {
    cout << "\n=== " << titulo << " ===" << endl;
}

// =====================================================================
// TESTE 1: iniPesosHe — fan_in correto
// =====================================================================
void testeIniPesos() {
    secao("Inicialização de pesos He");
    
    // He: E[w²] deve ser ~= 2/fan_in = 2/100 = 0.02
    // com fan_in = 100 entradas, fan_out = 10 saídas
    auto pesos = iniPesosHe(10, 100);
    
    verificar(pesos.size() == 10, "He: dimensão linhas correta (10)");
    verificar(pesos[0].size() == 100, "He: dimensão colunas correta (100)");
    
    // calcula variância empírica
    float soma = 0.0f, somaQ = 0.0f;
    int n = 0;
    for(const auto& linha : pesos) {
        for(float p : linha) { soma += p; somaQ += p * p; n++; }
    }
    float media = soma / n;
    float variancia = somaQ / n - media * media;
    float esperado = 2.0f / 100.0f; // 2/fan_in
    
    // tolerância generosa pra amostra aleatória (±50%)
    verificar(abs(variancia - esperado) < esperado * 0.5f, 
              "He: variância ~= 2/fan_in (" + to_string(variancia) + " ~= " + to_string(esperado) + ")");
    
    secao("Inicialização de pesos Xavier");
    auto pesosX = iniPesosXavier(50, 50);
    float somaX = 0.0f, somaQX = 0.0f;
    int nX = 0;
    for(const auto& linha : pesosX) {
        for(float p : linha) { somaX += p; somaQX += p * p; nX++; }
    }
    float varX = somaQX / nX - (somaX/nX) * (somaX/nX);
    // Xavier uniforme: variância = (2*limite)²/12 = limite²/3 = (6/(l+c))/3 = 1/(l+c)*2 ~= 0.02
    float esperadoX = 6.0f / (3.0f * (50.0f + 50.0f)); // var de uniforme(-lim, lim) = lim²/3
    verificar(abs(varX - esperadoX) < esperadoX * 0.5f,
              "Xavier: variância razoável (" + to_string(varX) + " ~= " + to_string(esperadoX) + ")");
}

// =====================================================================
// TESTE 2: Adam sem L2 oculto
// =====================================================================
void testeAdamSemL2() {
    secao("Adam — sem L2 oculto");
    
    // pesos zerados, gradiente constante
    // se tinha L2 antes, os pesos divergiam do esperado mesmo com grad fixo
    vector<vector<float>> pesos = {{1.0f, 1.0f}};
    vector<vector<float>> grad  = {{0.1f, 0.1f}};
    vector<float> bias = {1.0f};
    vector<float> gradBias = {0.1f};
    
    vector<vector<float>> m = {{0.0f, 0.0f}};
    vector<vector<float>> v = {{0.0f, 0.0f}};
    vector<float> mb = {0.0f};
    vector<float> vb = {0.0f};
    
    // 1 passo Adam, taxa=0.001, sem lambda
    float taxa = 0.001f, b1 = 0.9f, b2 = 0.999f, eps = 1e-8f;
    auto novos = attPesosAdam(pesos, grad, m, v, taxa, b1, b2, eps, 1);
    
    // cálculo manual: g=0.1, m=0.01, v=0.00001, mCorr=0.01/0.1=0.1, vCorr=0.00001/0.001=0.01
    // delta = 0.001 * 0.1 / (sqrt(0.01) + 1e-8) = 0.001 * 0.1 / 0.1 = 0.001
    float esperado = 1.0f - 0.001f;
    verificarPerto(novos[0][0], esperado, "Adam: passo correto sem L2", 1e-5f);
    
    // testa via classe Adam
    Adam adam(0.001f);
    vector<vector<float>> pesos2 = {{2.0f}};
    vector<vector<float>> grad2  = {{0.0f}}; // gradiente zero = pesos não devem mudar
    vector<float> bias2 = {0.0f};
    vector<float> gradBias2 = {0.0f};
    
    float pesosAntes = pesos2[0][0];
    adam.att(pesos2, grad2, bias2, gradBias2);
    // com grad=0 e sem L2: pesos ficam iguais (Adam com grad 0 não move nada)
    verificarPerto(pesos2[0][0], pesosAntes, "Adam: grad=0 não move pesos", 1e-6f);
}

// =====================================================================
// TESTE 3: AdamW sem L2 duplo
// =====================================================================
void testeAdamWCorreto() {
    secao("AdamW — decaimento de peso sem L2 duplo");
    
    AdamW adamw(0.001f, 0.01f); // taxa=0.001, pesoDecaimento=0.01
    vector<vector<float>> pesos = {{1.0f}};
    vector<vector<float>> grad  = {{0.0f}}; // grad zero pra isolar o decaimento
    vector<float> bias = {0.0f};
    vector<float> gradBias = {0.0f};
    
    adamw.att(pesos, grad, bias, gradBias);
    
    // com grad=0: só o decaimento age -> pesos *= (1 - taxa * pesoDecaimento) = 1 - 0.001*0.01 = 0.99999
    // depois o passo Adam com grad=0 move ~0
    float esperado = 1.0f * (1.0f - 0.001f * 0.01f); // ~0.99999
    verificarPerto(pesos[0][0], esperado, "AdamW: decaimento = (1 - taxa*wd) com grad=0", 1e-4f);
}

// =====================================================================
// TESTE 4: Dropout — retroprop não escala duas vezes
// =====================================================================
void testeDropout() {
    secao("Dropout — escalonamento correto na retropropagação");
    
    Dropout drop(0.5f, "drop", 42);
    drop.treinando = true;
    
    vector<float> entrada = {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f};
    auto saida = drop.prop(entrada);
    
    // na saída: valores são 0 ou 1/(1-0.5) = 2.0
    for(size_t i = 0; i < saida.size(); i++) {
        verificar(saida[i] == 0.0f || abs(saida[i] - 2.0f) < 1e-6f,
                  "Dropout prop: valor " + to_string(i) + " é 0 ou 2.0");
    }
    
    // retroprop com gradiente de saída = saída (simula forward pass por identidade)
    vector<float> gradSaida(saida.size(), 1.0f);
    auto gradEntrada = drop.retroprop(gradSaida);
    
    // FIX verificado: gradEntrada[i] deve ser 1.0 onde mascara=true, 0 onde mascara=false
    // NÃO deve ser 2.0 (que seria o double-scaling do bug original)
    for(size_t i = 0; i < gradEntrada.size(); i++) {
        verificar(gradEntrada[i] == 0.0f || abs(gradEntrada[i] - 1.0f) < 1e-6f,
                  "Dropout retroprop: gradiente " + to_string(i) + " é 0 ou 1.0 (não 2.0)");
    }
    
    // modo inferência: tudo passa
    drop.treinando = false;
    auto saidaTeste = drop.prop(entrada);
    verificar(saidaTeste == entrada, "Dropout inferência: saída = entrada");
}

// =====================================================================
// TESTE 5: Densa — gradientes corretos (verificação numérica)
// =====================================================================
void testeDensaGradientes() {
    secao("Densa — verificação numérica de gradientes");
    
    Densa densa(2, 2, "linear", true, "d1");
    // pesos fixos pra verificação determinística
    densa.defPesos({{0.5f, -0.3f}, {0.2f, 0.8f}});
    densa.defBias({0.1f, -0.1f});
    
    vector<float> entrada = {1.0f, 2.0f};
    auto saida = densa.prop(entrada);
    
    // z = W*x + b
    // z[0] = 0.5*1 + (-0.3)*2 + 0.1 = 0.5 - 0.6 + 0.1 = 0.0
    // z[1] = 0.2*1 + 0.8*2 + (-0.1) = 0.2 + 1.6 - 0.1 = 1.7
    verificarPerto(saida[0], 0.0f, "Densa prop: saida[0] = 0.0");
    verificarPerto(saida[1], 1.7f, "Densa prop: saida[1] = 1.7");
    
    // gradiente de saída = [1, 1] (MSE simplificado)
    vector<float> gradSaida = {1.0f, 1.0f};
    auto gradEntrada = densa.retroprop(gradSaida);
    
    // gradEntrada = W^T * gradSaida (ativação linear, derivada = 1)
    // gradEntrada[0] = W[0][0]*1 + W[1][0]*1 = 0.5 + 0.2 = 0.7
    // gradEntrada[1] = W[0][1]*1 + W[1][1]*1 = -0.3 + 0.8 = 0.5
    verificarPerto(gradEntrada[0], 0.7f, "Densa retroprop: gradEntrada[0] = 0.7");
    verificarPerto(gradEntrada[1], 0.5f, "Densa retroprop: gradEntrada[1] = 0.5");
    
    // gradPesos[i][j] = gradSaida[i] * entrada[j]
    // gradPesos[0][0] = 1*1 = 1, gradPesos[0][1] = 1*2 = 2
    // gradPesos[1][0] = 1*1 = 1, gradPesos[1][1] = 1*2 = 2
    verificarPerto(densa.gradPesos[0][0], 1.0f, "Densa: gradPesos[0][0] = 1.0");
    verificarPerto(densa.gradPesos[0][1], 2.0f, "Densa: gradPesos[0][1] = 2.0");
    verificarPerto(densa.gradBias[0], 1.0f,    "Densa: gradBias[0] = 1.0");
    verificarPerto(densa.gradBias[1], 1.0f,    "Densa: gradBias[1] = 1.0");
}

// =====================================================================
// TESTE 6: XOR — treino real
// =====================================================================
void testeXOR() {
    secao("XOR — treino completo");
    
    // dataset XOR
    vector<vector<float>> entradas = {{0,0},{0,1},{1,0},{1,1}};
    vector<vector<float>> alvos    = {{0},  {1},  {1},  {0}};
    
    Modelo modelo("xor");
    modelo.add(make_unique<Densa>(2, 8, "relu",    true, "d1"));
    modelo.add(make_unique<Densa>(8, 1, "sigmoid", true, "d2"));
    
    // Adam pra convergir rápido
    modelo.camadas[0]->defOtimizador(make_unique<Adam>(0.01f));
    modelo.camadas[1]->defOtimizador(make_unique<Adam>(0.01f));
    
    float ultimoErro = 1.0f;
    for(int epoca = 0; epoca < 2000; epoca++) {
        float erroTotal = 0.0f;
        modelo.zerarGradientes();
        
        for(size_t i = 0; i < entradas.size(); i++) {
            auto saida = modelo.prop(entradas[i]);
            erroTotal += mse(saida, alvos[i]);
            auto grad = derivadaMse(saida, alvos[i]);
            modelo.retroprop(grad);
        }
        // atualiza uma vez por época (acumulação de gradientes)
        modelo.att(0.01f);
        ultimoErro = erroTotal / entradas.size();
    }
    
    verificar(ultimoErro < 0.01f, "XOR: erro < 0.01 após 2000 épocas (" + to_string(ultimoErro) + ")");
    
    // verifica saídas individuais
    for(size_t i = 0; i < entradas.size(); i++) {
        auto saida = modelo.prop(entradas[i]);
        bool correto = (saida[0] > 0.5f) == (alvos[i][0] > 0.5f);
        verificar(correto, "XOR: classificação correta para entrada [" + 
                  to_string((int)entradas[i][0]) + "," + to_string((int)entradas[i][1]) + 
                  "] (saída=" + to_string(saida[0]) + ")");
    }
}

// =====================================================================
// TESTE 7: Classificação 4 classes (softmax + entropia cruzada)
// =====================================================================
void testeClassificacao4Classes() {
    secao("Classificação 4 classes — softmax + entropia cruzada");
    
    // 4 padrões lineares bem separados
    vector<vector<float>> entradas = {
        {1.0f, 0.0f, 0.0f, 0.0f},
        {0.0f, 1.0f, 0.0f, 0.0f},
        {0.0f, 0.0f, 1.0f, 0.0f},
        {0.0f, 0.0f, 0.0f, 1.0f}
    };
    // one-hot
    vector<vector<float>> alvos = {
        {1,0,0,0},
        {0,1,0,0},
        {0,0,1,0},
        {0,0,0,1}
    };
    
    Modelo modelo("clf4");
    modelo.add(make_unique<Densa>(4, 16, "relu",    true, "d1"));
    modelo.add(make_unique<Densa>(16, 4, "softmax", true, "d2"));
    
    modelo.camadas[0]->defOtimizador(make_unique<Adam>(0.005f));
    modelo.camadas[1]->defOtimizador(make_unique<Adam>(0.005f));
    
    float ultimoErro = 99.0f;
    for(int epoca = 0; epoca < 500; epoca++) {
        float erroTotal = 0.0f;
        modelo.zerarGradientes();
        
        for(size_t i = 0; i < entradas.size(); i++) {
            auto saida = modelo.prop(entradas[i]);
            erroTotal += entropiaCruzada(alvos[i], saida);
            auto grad = derivadaEntropiaCruzada(alvos[i], saida);
            modelo.retroprop(grad);
        }
        modelo.att(0.005f);
        ultimoErro = erroTotal / entradas.size();
    }
    
    verificar(ultimoErro < 0.1f, "4 classes: entropia cruzada < 0.1 (" + to_string(ultimoErro) + ")");
    
    // verifica acurácia
    int corretos = 0;
    for(size_t i = 0; i < entradas.size(); i++) {
        auto saida = modelo.prop(entradas[i]);
        if(argmax(saida) == argmax(alvos[i])) corretos++;
    }
    verificar(corretos == 4, "4 classes: 100% de acurácia (" + to_string(corretos) + "/4)");
}

// =====================================================================
// TESTE 8: LoteNorm — normalização correta
// =====================================================================
void testeLoteNorm() {
    secao("LoteNorm — normalização de lote");
    
    LoteNorm ln(3, 1e-5f, 0.9f, "ln1");
    ln.treinando = true;
    
    // lote com média e variância conhecidas por característica
    vector<vector<float>> lote = {
        {0.0f, 10.0f, -5.0f},
        {4.0f, 10.0f, -5.0f},
        {2.0f, 10.0f, -5.0f}
    };
    // característica 1: média=2, var=~2.67
    // característica 2: média=10, var=0 -> normalização dá 0 pra todos
    // característica 3: média=-5, var=0 -> idem
    
    auto saida = ln.propLote(lote);
    
    // com gamma=1, beta=0: saída normalizada
    // para coluna 0: valores são 0, 4, 2 -> media=2 -> normalizados: -1.22, +1.22, 0
    // saida[0][0] = -1.22 (min), saida[1][0] = +1.22 (max) -> soma ~= 0
    verificar(abs(saida[0][0] + saida[1][0]) < 1e-4f, "LoteNorm: simétrico ao redor da média");
    verificar(abs(saida[1][0]) > abs(saida[2][0]), "LoteNorm: ponto extremo tem valor maior");
    
    // com lote de variância 0: saída deve ser beta=0 (sem explodir)
    verificar(abs(saida[0][1]) < 0.1f, "LoteNorm: variância=0 não explode");
}

// =====================================================================
// TESTE 9: zerarGradientes funciona
// =====================================================================
void testeZerarGradientes() {
    secao("zerarGradientes — sem acumulação espúria entre épocas");
    
    Densa d(2, 2, "relu", true, "d");
    
    vector<float> e = {1.0f, 1.0f};
    vector<float> g = {1.0f, 1.0f};
    
    d.prop(e);
    d.retroprop(g);
    
    // gradientes têm valores
    float gradAntes = d.gradPesos[0][0];
    
    d.zerarGradientes();
    
    verificarPerto(d.gradPesos[0][0], 0.0f, "zerarGradientes: gradPesos zerado");
    verificarPerto(d.gradBias[0],     0.0f, "zerarGradientes: gradBias zerado");
    
    // segunda passagem sem zerar NÃO deve dobrar
    d.prop(e);
    d.retroprop(g);
    float gradDepois = d.gradPesos[0][0];
    
    // zera e faz de novo
    d.zerarGradientes();
    d.prop(e);
    d.retroprop(g);
    float gradLimpo = d.gradPesos[0][0];
    
    verificarPerto(gradDepois, gradLimpo, "zerarGradientes: gradiente idêntico após zerar");
}

// =====================================================================
// MAIN
// =====================================================================
int main() {
    cout << "=====================================================" << endl;
    cout << "  TESTES DA BIBLIS DE REDES NEURAIS" << endl;
    cout << "=====================================================" << endl;
    
    testeIniPesos();
    testeAdamSemL2();
    testeAdamWCorreto();
    testeDropout();
    testeDensaGradientes();
    testeXOR();
    testeClassificacao4Classes();
    testeLoteNorm();
    testeZerarGradientes();
    
    cout << "\n=====================================================" << endl;
    cout << "  RESULTADO: " << testesPassados << "/" << totalTestes << " testes passaram" << endl;
    if(testesPassados == totalTestes) {
        cout << "  TUDO OK" << endl;
    } else {
        cout << "  " << (totalTestes - testesPassados) << " FALHARAM" << endl;
    }
    cout << "=====================================================" << endl;
    
    return (testesPassados == totalTestes) ? 0 : 1;
}