// teste.cpp
#include "biblis/camadas.h"
#include <cassert>
#include <iomanip>

// teste pra camada densa
void testeCD1() {
    cout << "=== TESTE CAMADA DENSA ===\n\n";

    // teste 1: camada linear simples
    cout << "1. Teste camada linear (sem ativação):\n";
    Densa camada1(3, 2, "linear");

    // define pesos manualmente pra teste deterministico
    camada1.defPesos({{0.1f, 0.2f, 0.3f}, {0.4f, 0.5f, 0.6f}});
    camada1.defBias({0.1f, 0.2f});

    vector<float> entrada = {1.0f, 2.0f, 3.0f};
    vector<float> saida = camada1.prop(entrada);

    cout << "Entrada: ";
    for (float x : entrada) cout << x << " ";
    cout << "\nSaída: ";
    for (float x : saida) cout << x << " ";
    cout << "Esperado: 1.5 3.4\n\n";

    // teste 2: retropropagação
    cout << "2. Teste retropropagação:\n";
    vector<float> gradSaida = {0.1f, -0.2f};
    vector<float> gradEntrada = camada1.retroprop(gradSaida);

    cout << "Gradiente de entrada: ";
    for (float x : gradEntrada) cout << x << " ";
    cout << "\n\n";

    // teste 3: atualização de pesos
    cout << "3. Teste atualização de pesos:\n";
    camada1.att(0.01f);
    camada1.zerarGradientes();
    cout << "Pesos atualizados e gradientes zerados.\n\n";

    // teste 4: camada com ativação sigmoid
    cout << "4. Teste camada com sigmoid:\n";
    Densa camada2(2, 2, "sigmoid");
    camada2.defPesos({{0.5f, -0.3f}, {0.2f, 0.8f}});
    camada2.defBias({0.1f, -0.1f});

    vector<float> entrada2 = {0.5f, 0.8f};
    vector<float> saida2 = camada2.prop(entrada2);

    cout << "Entrada: ";
    for (float x : entrada2) cout << x << " ";
    cout << "\nSaída (sigmoid): ";
    for (float x : saida2) cout << x << " ";
    cout << "\n\n";

    // teste 5: processamento em lote
    cout << "5. Teste processamento em lote:\n";
    Densa camada3(2, 1, "relu");
    vector<vector<float>> lote = {
        {1.0f, 2.0f},
        {3.0f, 4.0f},
        {5.0f, 6.0f}
    };
    auto saidaLote = camada3.propLote(lote);
    cout << "Lote com " << saidaLote.size() << " exemplos processados.\n";

    // teste 6: informações da camada
    cout << "\n6. Informações da camada:\n";
    cout << "Tipo: " << camada1.tipo << endl;
    cout << "Tem parâmetros: " << (camada1.temParametros() ? "Sim" : "Não") << endl;
    cout << "Número de parâmetros: " << camada1.numParametros() << endl;
}

void testeCD2() {
    // teste 7: camada sem bias
    cout << "7. Teste camada SEM bias:\n";
    Densa camadaSemBias(2, 3, "linear", false);
    camadaSemBias.defPesos({{0.1f, 0.2f}, {0.3f, 0.4f}, {0.5f, 0.6f}});

    vector<float> entrada7 = {1.0f, 2.0f};
    vector<float> saida7 = camadaSemBias.prop(entrada7);

    cout << "Entrada: ";
    for (float x : entrada7) cout << x << " ";
    cout << "\nSaída (sem bias): ";
    for (float x : saida7) cout << x << " ";
    cout << "\nEsperado: 0.5 1.1 1.7\n\n";

    assert(fabs(saida7[0] - 0.5f) < 1e-6);
    assert(fabs(saida7[1] - 1.1f) < 1e-6);
    assert(fabs(saida7[2] - 1.7f) < 1e-6);

    // teste 8: diferentes funções de ativação
    cout << "8. Teste diferentes funções de ativação:\n";

    Densa camadaReLU(2, 2, "relu");
    camadaReLU.defPesos({{1.0f, -1.0f}, {0.5f, 0.5f}});
    camadaReLU.defBias({-0.5f, 0.0f});

    vector<float> entrada8 = {1.0f, -1.0f};
    vector<float> saidaReLU = camadaReLU.prop(entrada8);
    cout << "ReLU entrada [" << entrada8[0] << ", " << entrada8[1] << "]: ";
    cout << "[" << saidaReLU[0] << ", " << saidaReLU[1] << "]\n";

    Densa camadaTanh(2, 1, "tanh");
    camadaTanh.defPesos({{0.5f, -0.3f}});
    camadaTanh.defBias({0.2f});

    vector<float> entradaTanh = {0.8f, 0.2f};
    vector<float> saidaTanh = camadaTanh.prop(entradaTanh);
    cout << "Tanh entrada [" << entradaTanh[0] << ", " << entradaTanh[1] << "]: ";
    cout << "[" << saidaTanh[0] << "]\n\n";

    // teste 9: softmax
    cout << "9. Teste camada com softmax:\n";
    Densa camadaSoftmax(3, 3, "softmax");
    camadaSoftmax.defPesos({
        {0.1f, 0.2f, 0.3f},
        {0.4f, 0.5f, 0.6f},
        {0.7f, 0.8f, 0.9f}
    });
    camadaSoftmax.defBias({0.1f, 0.2f, 0.3f});

    vector<float> entrada9 = {1.0f, 0.5f, 0.2f};
    vector<float> saidaSoftmax = camadaSoftmax.prop(entrada9);

    cout << "Entrada: ";
    for (float x : entrada9) cout << x << " ";
    cout << "\nSaída softmax: ";
    for (float x : saidaSoftmax) cout << fixed << setprecision(4) << x << " ";

    float soma = 0.0f;
    for (float x : saidaSoftmax) soma += x;
    cout << "\nSoma: " << soma << " (deve ser ~1.0)\n\n";
    assert(fabs(soma - 1.0f) < 1e-4);
}

void testeCD() {
    cout << fixed << setprecision(4);
    testeCD1();
    testeCD2();
    cout << "\n=== FIM CAMADA DENSA ===\n";
}

int main() {
    testeCD();
    return 0;
}