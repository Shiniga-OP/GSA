// teste_xor.cpp
#include <iostream>
#include <vector>
#include <iomanip>
#include "biblis/camadas.h"

using namespace std;

int main() {
    // 1. dados do XOR
    // entradas:
    vector<vector<float>> entradas = {{0,0}, {0,1}, {1,0}, {1,1}};
    // saidas esperadas:
    vector<vector<float>> esperados = {{0}, {1}, {1}, {0}};

    // 2. define a arquitetura da rede
    // camada oculta: 2 entradas -> 3 neuronios
    Densa camadaOculta(2, 3, "relu");
    // camada de saida: 3 entradas -> 1 neuronio
    Densa camadaSaida(3, 1, "sigmoid");
    
    camadaOculta.defOtimizador(make_unique<Adam>(0.1f)); 
    camadaSaida.defOtimizador(make_unique<Adam>(0.1f));
    
    float taxaAprendizado = 0.5f;
    int epocas = 10000;

    cout << "Iniciando treinamento do XOR..." << endl;

    // 3. loop de Treinamento
    for(int i = 0; i < epocas; ++i) {
        float erroTotal = 0;

        for(size_t j = 0; j < entradas.size(); ++j) {
            // propagação:
            vector<float> saidaOculta = camadaOculta.prop(entradas[j]);
            vector<float> saidaFinal = camadaSaida.prop(saidaOculta);

            erroTotal += mse(saidaFinal, esperados[j]);

            // retropropagação, acumulando gradientes(treinamento por lote)
            vector<float> gradSaida = derivadaErro(saidaFinal, esperados[j]);
            vector<float> gradOculta = camadaSaida.retroprop(gradSaida);
            camadaOculta.retroprop(gradOculta);
        }
        // atualiza(apos ver todos os 4 exemplos)
        camadaSaida.att(taxaAprendizado); 
        camadaOculta.att(taxaAprendizado);

        // zera os gradientes acumulados do lote
        camadaSaida.zerarGradientes();
        camadaOculta.zerarGradientes();

        if(i % 2000 == 0) cout << "Epoca " << i << " - Erro Medio: " << erroTotal / 4 << endl;
    }
    // 4. teste Final
    cout << "\nResultados Finais:" << endl;
    cout << "A B | Saida | Esperado" << endl;
    cout << "-----------------------" << endl;
    for(size_t i = 0; i < entradas.size(); ++i) {
        vector<float> saida = camadaSaida.prop(camadaOculta.prop(entradas[i]));
        cout << entradas[i][0] << " " << entradas[i][1] << " | " 
             << fixed << setprecision(4) << saida[0] << " | " << esperados[i][0] << endl;
    }
    return 0;
}